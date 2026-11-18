"""Cleanup actions are validated and subprocess calls are mocked."""

from __future__ import annotations

import io
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

import ryoku_ai.cli as cli
from ryoku_ai.cli import Runtime
from ryoku_ai.config import Config, Credentials
from ryoku_ai.conversation import ConversationSession
from ryoku_ai.execution.approval import ApprovalEngine
from ryoku_ai.execution.cleanup import (
    CleanupActionValidator,
    CleanupExecutor,
    operation_spec,
)
from ryoku_ai.execution.models import ProposedAction, ValidationError
from ryoku_ai.execution.validator import CommandValidator
from ryoku_ai.storage.history import HistoryService


def make_proposed(operation: str) -> ProposedAction:
    spec = operation_spec(operation)
    return ProposedAction(
        title=spec.title,
        description=spec.expected_effect,
        executable=spec.executable,
        arguments=list(spec.arguments),
        required_privileges=list(spec.required_privileges),
        risk_level=spec.risk_level,
        expected_effect=spec.expected_effect,
        rollback_available=False,
    )


def make_cleanup_validator() -> CleanupActionValidator:
    return CleanupActionValidator(resolver=lambda name: f"/usr/bin/{name}")


def make_runner(calls: list[tuple[str, ...]], *, output: str = ""):
    def runner(argv, timeout):
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, output, "")

    return runner


def test_invalid_plan_schema_is_rejected() -> None:
    with pytest.raises(ValueError):
        ProposedAction.model_validate(
            {
                "title": "bad",
                "description": "bad",
                "executable": "rm",
                "arguments": ["-rf", "/"],
                "required_privileges": [],
                "risk_level": "critical",
                "expected_effect": "remove everything",
                "rollback_available": True,
            }
        )


def test_cleanup_rejects_shell_and_unlisted_commands() -> None:
    validator = make_cleanup_validator()
    dangerous = ProposedAction(
        title="cleanup",
        description="delete",
        executable="sh",
        arguments=["-c", "rm -rf /"],
        required_privileges=["root"],
        risk_level="high",
        expected_effect="delete files",
        rollback_available=False,
    )
    with pytest.raises(ValidationError, match="белом списке"):
        validator.validate(dangerous)

    forbidden_pacman = ProposedAction(
        title="clear all cache",
        description="dangerous",
        executable="pacman",
        arguments=["-Scc"],
        required_privileges=["root"],
        risk_level="high",
        expected_effect="delete everything",
        rollback_available=False,
    )
    with pytest.raises(ValidationError, match="белом списке"):
        validator.validate(forbidden_pacman)


def test_approval_metadata_comes_from_allowlist_not_model_text() -> None:
    validator = make_cleanup_validator()
    proposed = make_proposed("clean_pacman_cache").model_copy(
        update={
            "title": "Ничего не удалять",
            "description": "Безопасно, без последствий",
            "expected_effect": "Никакие файлы не изменятся",
        }
    )

    validated = validator.validate(proposed)

    assert validated.title == operation_spec("clean_pacman_cache").title
    assert validated.expected_effect == operation_spec("clean_pacman_cache").expected_effect


def test_approval_panel_confirms_one_action_and_cancel_stops_plan() -> None:
    action_validator = make_cleanup_validator()
    action = action_validator.validate(make_proposed("inspect_user_cache"))
    output = io.StringIO()
    engine = ApprovalEngine(
        Console(file=output, force_terminal=False, width=100),
        CommandValidator(resolver=lambda name: f"/usr/bin/{name}"),
        input_fn=lambda _prompt: "",
        interactive=True,
    )

    approved = engine.request_action_approval(action, action_validator, index=1, total=2)

    assert approved.approved is True
    assert "RYOKU AI — ACTION APPROVAL" in output.getvalue()
    assert "[Enter] Подтвердить" in output.getvalue()

    cancelled_engine = ApprovalEngine(
        Console(file=io.StringIO(), force_terminal=False),
        CommandValidator(resolver=lambda name: f"/usr/bin/{name}"),
        input_fn=lambda _prompt: "4",
        interactive=True,
    )
    cancelled = cancelled_engine.request_action_approval(
        action, action_validator, index=1, total=2
    )
    assert cancelled.approved is False
    assert cancelled.stop_plan is True


def test_root_action_requires_second_confirmation() -> None:
    action_validator = make_cleanup_validator()
    action = action_validator.validate(make_proposed("inspect_journal"))
    answers = iter(["", "2"])
    engine = ApprovalEngine(
        Console(file=io.StringIO(), force_terminal=False),
        CommandValidator(resolver=lambda name: f"/usr/bin/{name}"),
        input_fn=lambda _prompt: next(answers),
        interactive=True,
    )

    outcome = engine.request_action_approval(action, action_validator, index=1, total=1)

    assert outcome.approved is False
    assert outcome.reason == "sudo отменён"


def test_edit_action_restricts_to_allowlist_and_refreshes_metadata() -> None:
    action_validator = make_cleanup_validator()
    action = action_validator.validate(make_proposed("clean_tmp"))
    answers = iter(["2", "paccache -r -k 3", "", "1"])
    engine = ApprovalEngine(
        Console(file=io.StringIO(), force_terminal=False),
        CommandValidator(resolver=lambda name: f"/usr/bin/{name}"),
        input_fn=lambda _prompt: next(answers),
        interactive=True,
    )

    outcome = engine.request_action_approval(action, action_validator, index=1, total=1)

    assert outcome.approved is True
    assert outcome.action.operation == "clean_pacman_cache"
    assert outcome.action.title == operation_spec("clean_pacman_cache").title


def test_changed_arguments_after_confirmation_are_rejected(monkeypatch) -> None:
    validator = make_cleanup_validator()
    action = validator.validate(make_proposed("inspect_user_cache"))
    changed = replace(action, arguments=("-rf", "/home/user/.cache"))
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "ryoku_ai.execution.cleanup.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=1000),
    )
    executor = CleanupExecutor(validator, runner=make_runner(calls))

    result = executor.execute(changed)

    assert result.error is not None
    assert "изменилось" in result.error
    assert calls == []


def test_user_cache_is_analysis_only(monkeypatch) -> None:
    validator = make_cleanup_validator()
    action = validator.validate(make_proposed("inspect_user_cache"))
    calls: list[tuple[str, ...]] = []
    home = Path("/tmp/test-home")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(
        "ryoku_ai.execution.cleanup.inspect_directory",
        lambda path: SimpleNamespace(
            as_dict=lambda: {"path": str(path), "status": "success", "readable_size": "12M"},
            status="success",
            summary="Доступный размер: 12M.",
        ),
    )
    monkeypatch.setattr(
        "ryoku_ai.execution.cleanup.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=1000),
    )
    executor = CleanupExecutor(validator, runner=make_runner(calls))

    result = executor.execute(action)

    assert result.ok is True
    assert calls == []
    assert str(home / ".cache") in result.stdout


def test_cleanup_requires_successful_inspection(monkeypatch) -> None:
    validator = make_cleanup_validator()
    action = validator.validate(make_proposed("clean_pacman_cache"))
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "ryoku_ai.execution.cleanup.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=1000),
    )
    result = CleanupExecutor(validator, runner=make_runner(calls)).execute(
        action,
        sudo_approved=True,
    )

    assert result.ok is False
    assert "обязательный анализ" in result.error
    assert calls == []


def test_journal_vacuum_requires_prior_disk_usage(monkeypatch) -> None:
    validator = make_cleanup_validator()
    action = validator.validate(make_proposed("vacuum_journal"))
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "ryoku_ai.execution.cleanup.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=1000),
    )

    result = CleanupExecutor(validator, runner=make_runner(calls)).execute(
        action,
        sudo_approved=True,
    )

    assert result.ok is False
    assert calls == []


def test_subprocess_oserror_and_timeout_are_reported(monkeypatch) -> None:
    validator = make_cleanup_validator()
    action = validator.validate(make_proposed("inspect_pacman_cache"))
    monkeypatch.setattr(
        "ryoku_ai.execution.cleanup.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=1000),
    )

    def fail(_argv, _timeout):
        raise OSError("mocked failure")

    result = CleanupExecutor(validator, runner=fail).execute(action, sudo_approved=True)
    assert result.ok is False
    assert "Не удалось запустить" in result.error

    def timeout(_argv, _timeout):
        raise subprocess.TimeoutExpired("du", 1)

    result = CleanupExecutor(validator, runner=timeout).execute(action, sudo_approved=True)
    assert result.timed_out is True
    assert "Таймаут" in result.error


def test_tmp_cleanup_refuses_open_files(monkeypatch) -> None:
    validator = make_cleanup_validator()
    action = validator.validate(make_proposed("clean_tmp"))
    calls: list[tuple[str, ...]] = []

    def runner(argv, timeout):
        calls.append(tuple(argv))
        if any(argument.endswith("/lsof") for argument in argv):
            return subprocess.CompletedProcess(argv, 0, "1234\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(
        "ryoku_ai.execution.cleanup.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=1000),
    )
    result = CleanupExecutor(validator, runner=runner).execute(
        action,
        sudo_approved=True,
        completed_inspections={"inspect_tmp"},
    )

    assert result.ok is False
    assert "открытые или используемые" in result.error
    assert any(any(argument.endswith("/lsof") for argument in call) for call in calls)
    assert not any(call[0].endswith("systemd-tmpfiles") for call in calls)


def test_cli_action_plan_renders_approval_and_feedback(monkeypatch, tmp_path) -> None:
    config = Config()
    console_output = io.StringIO()
    console = Console(file=console_output, force_terminal=False, width=100)
    history = HistoryService(tmp_path / "history.db", enabled=True, secrets=[])
    runtime = Runtime(config, Credentials(), console, history, [])
    session = ConversationSession(config, history, title="кеш")
    session.add_user("Проанализируй пользовательский кеш")
    action_validator = make_cleanup_validator()

    class FakeCleanupExecutor:
        def __init__(self, _validator, *, timeout):
            self.timeout = timeout

        def execute(self, action, **_kwargs):
            return cli.CleanupResult(
                action.operation,
                action.display(),
                0,
                stdout="12M /home/test/.cache",
                stderr="notice",
                free_before=100,
                free_after=180,
            )

    monkeypatch.setattr(cli, "CleanupActionValidator", lambda: action_validator)
    monkeypatch.setattr(cli, "CleanupExecutor", FakeCleanupExecutor)
    monkeypatch.setattr(
        cli,
        "ApprovalEngine",
        lambda terminal, validator, allow_sudo: ApprovalEngine(
            terminal,
            validator,
            allow_sudo=allow_sudo,
            input_fn=lambda _prompt: "",
            interactive=True,
        ),
    )

    cli._run_commands(
        runtime,
        session,
        cli.AIAction(actions=[make_proposed("inspect_user_cache")]),
    )

    output = console_output.getvalue()
    assert "RYOKU AI — ACTION PLAN" in output
    assert "Проанализируй пользовательский кеш" in output
    assert "[1] Анализ пользовательского кеша" in output
    assert "RYOKU AI — ACTION APPROVAL" in output
    assert "ACTION RESULT — SUCCESS" in output
    assert "Код завершения: 0" in output
    assert "stdout:" in output
    assert "stderr:" in output
    assert "свободное место" in output.lower()

    cli._show_history(runtime, show_id=session.conversation_id)
    assert "Полный вывод" in console_output.getvalue()
    assert "12M /home/test/.cache" in console_output.getvalue()


def test_no_exec_renders_cleanup_and_command_plans(monkeypatch, tmp_path) -> None:
    config = Config()
    console_output = io.StringIO()
    console = Console(file=console_output, force_terminal=False, width=100)
    history = HistoryService(tmp_path / "history.db", enabled=True, secrets=[])
    runtime = Runtime(config, Credentials(), console, history, [])
    session = ConversationSession(config, history, title="dry run")
    action_validator = make_cleanup_validator()
    command_validator = CommandValidator(resolver=lambda name: f"/usr/bin/{name}")

    class ForbiddenExecutor:
        def __init__(self, *_args, **_kwargs):
            pass

        def execute(self, *_args, **_kwargs):
            raise AssertionError("--no-exec must never run cleanup")

    monkeypatch.setattr(cli, "CleanupActionValidator", lambda: action_validator)
    monkeypatch.setattr(cli, "CleanupExecutor", ForbiddenExecutor)
    monkeypatch.setattr(cli, "CommandValidator", lambda **_kwargs: command_validator)

    cli._run_commands(
        runtime,
        session,
        cli.AIAction(
            actions=[make_proposed("inspect_user_cache")],
            commands=[cli.ProposedCommand(argv=["ps", "aux"], description="Processes")],
        ),
        allow_exec=False,
    )

    output = console_output.getvalue()
    assert "RYOKU AI — ACTION PLAN" in output
    assert "ПЛАН (выполнение отключено)" in output


def test_cleanup_result_reports_partial_inspection(monkeypatch, tmp_path) -> None:
    config = Config()
    console_output = io.StringIO()
    console = Console(file=console_output, force_terminal=False, width=100)
    history = HistoryService(tmp_path / "history.db", enabled=True, secrets=[])
    runtime = Runtime(config, Credentials(), console, history, [])
    session = ConversationSession(config, history, title="tmp")
    action_validator = make_cleanup_validator()

    class PartialCleanupExecutor:
        def __init__(self, _validator, *, timeout):
            pass

        def execute(self, action, **_kwargs):
            return cli.CleanupResult(
                action.operation,
                action.display(),
                1,
                stdout='[{"path":"/tmp","status":"partial_success","readable_size":"39 MiB","inaccessible_directories":3,"warnings":["permission denied"]}]',
            )

    monkeypatch.setattr(cli, "CleanupActionValidator", lambda: action_validator)
    monkeypatch.setattr(cli, "CleanupExecutor", PartialCleanupExecutor)
    monkeypatch.setattr(
        cli,
        "ApprovalEngine",
        lambda terminal, validator, allow_sudo: ApprovalEngine(
            terminal,
            validator,
            allow_sudo=allow_sudo,
            input_fn=lambda _prompt: "",
            interactive=True,
        ),
    )

    cli._run_commands(
        runtime,
        session,
        cli.AIAction(actions=[make_proposed("inspect_tmp")]),
    )

    output = console_output.getvalue()
    assert "ACTION RESULT — PARTIAL SUCCESS" in output
    assert "39 MiB" in output
    assert "Недоступные каталоги: 3" in output
