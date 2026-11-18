"""Исполнитель: argv, таймауты, лимиты вывода и повторная проверка."""

from __future__ import annotations

import sys
from dataclasses import replace
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from ryoku_ai.cli import _render_execution
from ryoku_ai.config import Config
from ryoku_ai.execution.executor import ExecutionResult, Executor, safe_env
from ryoku_ai.execution.models import ValidatedCommand, ValidationError, compute_digest
from ryoku_ai.execution.validator import CommandValidator


def python_command(code: str, *, requires_sudo: bool = False) -> ValidatedCommand:
    argv = tuple(["python", "-c", code])
    executable = sys.executable
    return ValidatedCommand(
        argv=argv,
        executable=executable,
        description="тест",
        risk="low",
        requires_sudo=requires_sudo,
        policy_note="тест",
        digest=compute_digest(
            argv=argv,
            executable=executable,
            requires_sudo=requires_sudo,
        ),
    )


def identity(command: ValidatedCommand) -> ValidatedCommand:
    return command


def test_successful_execution_captures_stdout_stderr() -> None:
    command = python_command("import sys; print('hello'); print('err', file=sys.stderr)")
    executor = Executor(Config(), revalidator=identity)
    result = executor.execute(command)
    assert result.ok is True
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"
    assert result.stderr.strip() == "err"
    assert result.timed_out is False
    assert result.truncated is False
    assert "[stderr]" in result.combined


def test_nonzero_exit_code_is_reported() -> None:
    command = python_command("import sys; print('x'); sys.exit(7)")
    result = Executor(Config(), revalidator=identity).execute(command)
    assert result.ok is False
    assert result.exit_code == 7
    assert result.stdout.strip() == "x"
    assert result.status == "failed"


def test_timeout_kills_process() -> None:
    command = python_command("import time; time.sleep(5)")
    result = Executor(Config(), revalidator=identity).execute(command, timeout=1.0)
    assert result.timed_out is True
    assert result.exit_code is None
    assert result.status == "timeout"
    assert result.duration_ms >= 900


def test_output_is_truncated_to_limit() -> None:
    command = python_command("import sys; sys.stdout.write('a' * 5000)")
    result = Executor(Config(max_output_chars=1000), revalidator=identity).execute(
        command,
        max_output_chars=1000,
    )
    assert result.truncated is True
    assert len(result.stdout) <= 1000
    assert set(result.stdout) == {"a"}


def test_stderr_is_truncated_to_limit() -> None:
    command = python_command("import sys; sys.stderr.write('b' * 5000)")
    result = Executor(Config(max_output_chars=1000), revalidator=identity).execute(
        command,
        max_output_chars=1000,
    )
    assert result.truncated is True
    assert len(result.stderr) <= 1000


def test_safe_environment_has_no_extra_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOP_SECRET_VALUE", "do-not-pass")
    command = python_command(
        "import os; print(os.environ.get('TOP_SECRET_VALUE', 'missing')); "
        "print(os.environ.get('PATH'))"
    )
    result = Executor(Config(), revalidator=identity).execute(command)
    lines = result.stdout.splitlines()
    assert lines[0] == "missing"
    assert lines[1] == "/usr/local/bin:/usr/bin:/bin"


def test_missing_workdir_is_rejected(tmp_path: Path) -> None:
    command = python_command("print('x')")
    result = Executor(Config(), revalidator=identity).execute(
        command,
        cwd=tmp_path / "missing",
    )
    assert result.error is not None
    assert "Рабочий каталог" in result.error


def test_revalidation_error_prevents_execution() -> None:
    def reject(_command: ValidatedCommand) -> ValidatedCommand:
        raise ValidationError("изменилось")

    command = python_command("print('should not run')")
    result = Executor(Config(), revalidator=reject).execute(command)
    assert result.exit_code is None
    assert result.error is not None
    assert "Проверка перед запуском" in result.error
    assert result.stdout == ""


def test_digest_mismatch_prevents_execution() -> None:
    command = python_command("print('should not run')")
    result = Executor(
        Config(),
        revalidator=lambda item: replace(item, digest="0" * 64),
    ).execute(command)
    assert result.exit_code is None
    assert result.error == "Команда изменилась после подтверждения — запуск отменён"


def test_missing_executable_is_reported(tmp_path: Path) -> None:
    argv = ("missing",)
    executable = str(tmp_path / "missing")
    command = ValidatedCommand(
        argv=argv,
        executable=executable,
        description="",
        risk="low",
        requires_sudo=False,
        policy_note="",
        digest=compute_digest(
            argv=argv,
            executable=executable,
            requires_sudo=False,
        ),
    )
    result = Executor(Config(), revalidator=identity).execute(command)
    assert result.exit_code is None
    assert result.error is not None
    assert "Не удалось запустить" in result.error


def test_sudo_missing_is_reported_without_running() -> None:
    argv = ("ss", "-tunap")
    command = ValidatedCommand(
        argv=argv,
        executable="/usr/bin/ss",
        description="",
        risk="medium",
        requires_sudo=True,
        policy_note="",
        digest=compute_digest(
            argv=argv,
            executable="/usr/bin/ss",
            requires_sudo=True,
        ),
    )
    validator = CommandValidator(
        resolver=lambda name: None if name == "sudo" else "/usr/bin/ss"
    )
    result = Executor(
        Config(),
        validator=validator,
        revalidator=identity,
    ).execute(command)
    assert result.exit_code is None
    assert result.error is not None
    assert "sudo не найден" in result.error


def test_execution_result_properties() -> None:
    result = ExecutionResult(
        argv=("x",),
        exit_code=None,
        stdout="",
        stderr="",
        duration_ms=1,
        error="boom",
    )
    assert result.ok is False
    assert result.combined == "[error] boom"
    assert result.status == "failed"
    assert ExecutionResult.cancelled(("tool",)).status == "cancelled"


def test_execution_result_classifies_complete_success() -> None:
    result = ExecutionResult(
        argv=("du", "-sh", "/tmp"),
        exit_code=0,
        stdout="39M\t/tmp",
        stderr="",
        duration_ms=11,
    )

    assert result.status == "success"
    assert result.summary == "Команда выполнена успешно."
    assert result.structured()["exit_code"] == 0


def test_du_nonzero_with_size_and_permission_warnings_is_partial() -> None:
    result = ExecutionResult(
        argv=("du", "-sh", "/tmp"),
        exit_code=1,
        stdout="39M\t/tmp",
        stderr=(
            "du: cannot read directory '/tmp/private-a': Permission denied\n"
            "du: cannot read directory '/tmp/private-b': Permission denied"
        ),
        duration_ms=11,
    )

    assert result.status == "partial_success"
    assert result.exit_code == 1
    assert result.stdout == "39M\t/tmp"
    assert result.inaccessible_count == 2
    assert result.warnings == ["Недоступных каталогов: 2 (Permission denied)."]
    assert "приблизительной нижней оценкой" in result.summary


def test_permission_warnings_with_zero_exit_and_output_are_partial() -> None:
    result = ExecutionResult(
        argv=("du", "-sh", "/var/tmp"),
        exit_code=0,
        stdout="0B\t/var/tmp",
        stderr="du: cannot read directory '/var/tmp/private': Permission denied",
        duration_ms=3,
    )

    assert result.status == "partial_success"
    assert result.inaccessible_count == 1


def test_du_permission_failure_without_stdout_is_failed() -> None:
    result = ExecutionResult(
        argv=("du", "-sh", "/tmp"),
        exit_code=1,
        stdout="",
        stderr="du: cannot read directory '/tmp/private': Permission denied",
        duration_ms=3,
    )

    assert result.status == "failed"
    assert result.inaccessible_count == 1
    assert result.stderr


def test_partial_result_counts_distinct_inaccessible_paths() -> None:
    result = ExecutionResult(
        argv=("find", "/tmp"),
        exit_code=1,
        stdout="/tmp/readable",
        stderr=(
            "find: '/tmp/a': Permission denied\n"
            "find: '/tmp/b': Permission denied\n"
            "find: '/tmp/a': Permission denied"
        ),
        duration_ms=5,
    )

    assert result.status == "partial_success"
    assert result.inaccessible_count == 2
    assert "Недоступных путей: 2" in result.warnings[0]


def test_rich_renderer_summarizes_permission_stderr_and_keeps_stdout() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=80)
    command = python_command("print('unused')")
    result = ExecutionResult(
        argv=("du", "-sh", "/tmp"),
        exit_code=1,
        stdout="39 MB\t/tmp",
        stderr="\n".join(
            f"du: cannot read directory '/tmp/private-{index}': Permission denied"
            for index in range(10)
        ),
        duration_ms=11,
    )

    _render_execution(console, command, result, result.combined)

    rendered = output.getvalue()
    assert "PARTIAL SUCCESS" in rendered
    assert "39 MB" in rendered
    assert "Недоступных каталогов: 10" in rendered
    assert "/tmp/private-0" not in rendered
    assert "10 permission warning(s) summarized" in rendered


def test_safe_env_contains_expected_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    env = safe_env()
    assert env["PATH"] == "/usr/local/bin:/usr/bin:/bin"
    assert env["TERM"] == "dumb"
    assert env["NO_COLOR"] == "1"
    assert env["XDG_RUNTIME_DIR"] == "/run/user/1000"
    assert "TOP_SECRET_VALUE" not in env
