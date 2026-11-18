"""Allowlisted cleanup actions with preflight checks and no shell execution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..execution.models import ProposedAction, RiskLevel, ValidationError
from ..execution.policy import SAFE_EXEC_DIRS
from ..system.temp_inspector import inspect_directory
from .executor import _LimitedReader, safe_env


@dataclass(frozen=True, slots=True)
class CleanupSpec:
    operation: str
    title: str
    executable: str
    arguments: tuple[str, ...]
    required_privileges: tuple[str, ...]
    risk_level: RiskLevel
    expected_effect: str
    prerequisite: str | None = None

    @property
    def requires_sudo(self) -> bool:
        return "root" in self.required_privileges


_SPECS = (
    CleanupSpec(
        "inspect_tmp",
        "Анализ /tmp и /var/tmp",
        "ryoku-temp-inspector",
        ("/tmp", "/var/tmp"),
        (),
        "low",
        "Непривилегированно измерит доступные данные в /tmp и /var/tmp без перехода по symlink.",
    ),
    CleanupSpec(
        "clean_tmp",
        "Очистка просроченных временных файлов",
        "systemd-tmpfiles",
        ("--clean", "--prefix=/tmp"),
        ("root",),
        "medium",
        "Удалит только просроченные файлы по правилам systemd-tmpfiles.",
        prerequisite="inspect_tmp",
    ),
    CleanupSpec(
        "inspect_pacman_cache",
        "Анализ кеша pacman",
        "paccache",
        ("-d", "-k", "3"),
        ("root",),
        "low",
        "Покажет размер кеша и пакеты сверх трёх последних версий.",
    ),
    CleanupSpec(
        "clean_pacman_cache",
        "Безопасная очистка кеша pacman",
        "paccache",
        ("-r", "-k", "3"),
        ("root",),
        "medium",
        "Оставит три последние версии каждого пакета; pacman -Scc не используется.",
        prerequisite="inspect_pacman_cache",
    ),
    CleanupSpec(
        "inspect_user_cache",
        "Анализ пользовательского кеша",
        "ryoku-temp-inspector",
        ("~/.cache",),
        (),
        "low",
        "Покажет размер ~/.cache без удаления файлов.",
    ),
    CleanupSpec(
        "inspect_journal",
        "Анализ размера журналов systemd",
        "journalctl",
        ("--disk-usage",),
        ("root",),
        "low",
        "Покажет объём журналов; ничего не удалит.",
    ),
    CleanupSpec(
        "vacuum_journal",
        "Предложить ограничение архивных журналов до 500 МБ",
        "journalctl",
        ("--vacuum-size=500M",),
        ("root",),
        "high",
        "Удалит только архивные журналы сверх лимита; активные журналы не тронет.",
        prerequisite="inspect_journal",
    ),
)
_SPECS_BY_COMMAND = {
    (spec.executable, spec.arguments): spec
    for spec in _SPECS
}
_SPECS_BY_OPERATION = {spec.operation: spec for spec in _SPECS}


@dataclass(frozen=True, slots=True)
class ValidatedCleanupAction:
    operation: str
    title: str
    description: str
    executable: str
    arguments: tuple[str, ...]
    required_privileges: tuple[str, ...]
    risk_level: RiskLevel
    expected_effect: str
    rollback_available: bool
    prerequisite: str | None
    digest: str

    @property
    def requires_sudo(self) -> bool:
        return "root" in self.required_privileges

    def display(self) -> str:
        import shlex

        return shlex.join((self.executable, *self.arguments))


class CleanupActionValidator:
    def __init__(
        self,
        *,
        resolver: Callable[[str], str | None] = shutil.which,
    ):
        self.resolver = resolver

    def _resolve(self, executable: str) -> str:
        path = self.resolver(executable)
        if not path:
            raise ValidationError(f"Не найдена утилита {executable}", category="missing-tool")
        resolved = Path(path).resolve()
        if not resolved.is_absolute() or str(resolved.parent) not in SAFE_EXEC_DIRS:
            raise ValidationError("Исполняемый файл находится вне системных каталогов")
        return str(resolved)

    @staticmethod
    def _digest(action: ProposedAction, executable: str) -> str:
        payload = {
            "title": action.title,
            "description": action.description,
            "executable": executable,
            "arguments": action.arguments,
            "required_privileges": action.required_privileges,
            "risk_level": action.risk_level,
            "expected_effect": action.expected_effect,
            "rollback_available": action.rollback_available,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate(self, action: ProposedAction) -> ValidatedCleanupAction:
        spec = _SPECS_BY_COMMAND.get((action.executable, tuple(action.arguments)))
        if spec is None:
            raise ValidationError("Действие отсутствует в белом списке безопасной очистки")
        if tuple(action.required_privileges) != spec.required_privileges:
            raise ValidationError("Уровень привилегий не соответствует операции")
        if action.risk_level != spec.risk_level:
            raise ValidationError("Уровень риска не соответствует операции")
        if action.rollback_available:
            raise ValidationError("Метаданные действия не соответствуют безопасной операции")
        executable = (
            spec.executable
            if spec.executable == "ryoku-temp-inspector"
            else self._resolve(spec.executable)
        )
        canonical = ProposedAction(
            title=spec.title,
            description=spec.expected_effect,
            executable=spec.executable,
            arguments=list(spec.arguments),
            required_privileges=list(spec.required_privileges),
            risk_level=spec.risk_level,
            expected_effect=spec.expected_effect,
            rollback_available=False,
        )
        return ValidatedCleanupAction(
            operation=spec.operation,
            title=spec.title,
            description=spec.expected_effect,
            executable=executable,
            arguments=spec.arguments,
            required_privileges=spec.required_privileges,
            risk_level=spec.risk_level,
            expected_effect=spec.expected_effect,
            rollback_available=False,
            prerequisite=spec.prerequisite,
            digest=self._digest(canonical, executable),
        )

    def proposed_for(self, executable: str, arguments: Sequence[str]) -> ProposedAction:
        spec = _SPECS_BY_COMMAND.get((executable, tuple(arguments)))
        if spec is None:
            raise ValidationError("Изменённая операция отсутствует в белом списке")
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

    def revalidate(self, action: ValidatedCleanupAction) -> ValidatedCleanupAction:
        spec = _SPECS_BY_OPERATION.get(action.operation)
        if spec is None:
            raise ValidationError("Неизвестная cleanup-операция")
        proposed = ProposedAction(
            title=action.title,
            description=action.description,
            executable=spec.executable,
            arguments=list(action.arguments),
            required_privileges=list(spec.required_privileges),
            risk_level=spec.risk_level,
            expected_effect=action.expected_effect,
            rollback_available=action.rollback_available,
        )
        fresh = self.validate(proposed)
        if fresh != action:
            raise ValidationError("Действие изменилось после подтверждения")
        return fresh

    def variants(self) -> tuple[ProposedAction, ...]:
        return tuple(
            ProposedAction(
                title=spec.title,
                description=spec.expected_effect,
                executable=spec.executable,
                arguments=list(spec.arguments),
                required_privileges=list(spec.required_privileges),
                risk_level=spec.risk_level,
                expected_effect=spec.expected_effect,
                rollback_available=False,
            )
            for spec in _SPECS
        )


@dataclass(slots=True)
class CleanupResult:
    operation: str
    command: str
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    timed_out: bool = False
    error: str | None = None
    free_before: int | None = None
    free_after: int | None = None

    @property
    def status(self) -> str:
        if self.timed_out:
            return "timeout"
        if self.exit_code == 0 and self.error is None:
            return "success"
        if self.exit_code not in (0, None) and self.stdout.strip() and self.error is None:
            return "partial_success"
        return "failed"

    @property
    def ok(self) -> bool:
        return self.status == "success"


Runner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]


def _run_process(argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        list(argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        env=safe_env(),
        close_fds=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_reader = _LimitedReader(process.stdout, 800_000)
    stderr_reader = _LimitedReader(process.stderr, 800_000)
    stdout_reader.start()
    stderr_reader.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait()
    stdout_reader.join(timeout=2)
    stderr_reader.join(timeout=2)
    stdout = stdout_reader.text()
    stderr = stderr_reader.text()
    if stdout_reader.truncated or stderr_reader.truncated:
        stderr += "\n[вывод усечён: превышен лимит 200000 символов]"
    if timed_out:
        raise subprocess.TimeoutExpired(list(argv), timeout, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(list(argv), process.returncode, stdout, stderr)


class CleanupExecutor:
    def __init__(
        self,
        validator: CleanupActionValidator,
        *,
        runner: Runner = _run_process,
        timeout: float = 30.0,
    ):
        self.validator = validator
        self.runner = runner
        self.timeout = max(1.0, min(timeout, 120.0))

    def _free_space(self, operation: str) -> int | None:
        targets = {
            "inspect_tmp": Path("/tmp"),
            "clean_tmp": Path("/tmp"),
            "inspect_pacman_cache": Path("/var/cache/pacman/pkg"),
            "clean_pacman_cache": Path("/var/cache/pacman/pkg"),
            "inspect_user_cache": Path.home() / ".cache",
            "inspect_journal": Path("/var/log"),
            "vacuum_journal": Path("/var/log"),
        }
        target = targets.get(operation)
        if target is None:
            return None
        if not target.exists():
            target = target.parent
        try:
            return shutil.disk_usage(target).free
        except OSError:
            return None

    def _command(self, name: str, arguments: Sequence[str], *, sudo: bool) -> tuple[str, ...]:
        path = self.validator._resolve(name)
        argv: tuple[str, ...] = (path, *arguments)
        if sudo:
            sudo_path = self.validator._resolve("sudo")
            argv = (sudo_path, "-n", "--", *argv)
        return argv

    def _invoke(self, name: str, arguments: Sequence[str], *, sudo: bool) -> subprocess.CompletedProcess[str]:
        return self.runner(self._command(name, arguments, sudo=sudo), self.timeout)

    def _attempt(
        self,
        name: str,
        arguments: Sequence[str],
        *,
        sudo: bool,
        stdout: list[str],
        stderr: list[str],
    ) -> tuple[int | None, str | None, bool]:
        try:
            result = self._invoke(name, arguments, sudo=sudo)
        except subprocess.TimeoutExpired as exc:
            if isinstance(exc.output, str) and exc.output:
                stdout.append(exc.output.rstrip())
            if isinstance(exc.stderr, str) and exc.stderr:
                stderr.append(exc.stderr.rstrip())
            return None, f"Таймаут команды {name}", True
        except OSError as exc:
            return None, f"Не удалось запустить {name}: {exc}", False
        if result.stdout:
            stdout.append(result.stdout.rstrip())
        if result.stderr:
            stderr.append(result.stderr.rstrip())
        if result.returncode != 0:
            return result.returncode, f"{name} завершился с кодом {result.returncode}", False
        return 0, None, False

    def _check_open_files(
        self,
        path: str,
        *,
        sudo: bool,
        stdout: list[str],
        stderr: list[str],
    ) -> str | None:
        try:
            result = self._invoke("lsof", ("-t", "+D", path), sudo=sudo)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"Не удалось проверить открытые файлы в {path}: {type(exc).__name__}"
        if result.stderr:
            stderr.append(result.stderr.rstrip())
            return f"Проверка открытых файлов в {path} завершилась с ошибкой"
        if result.stdout.strip():
            stdout.append(f"Открытые файлы в {path}:\n{result.stdout.rstrip()}")
            return f"Очистка {path} остановлена: найдены открытые или используемые файлы"
        if result.returncode not in {0, 1}:
            return f"Проверка открытых файлов в {path} завершилась с кодом {result.returncode}"
        return None

    def execute(
        self,
        action: ValidatedCleanupAction,
        *,
        sudo_approved: bool = False,
        completed_inspections: set[str] | None = None,
    ) -> CleanupResult:
        started = time.monotonic()
        stdout: list[str] = []
        stderr: list[str] = []
        before = self._free_space(action.operation)
        try:
            fresh = self.validator.revalidate(action)
        except ValidationError as exc:
            return CleanupResult(
                action.operation,
                action.display(),
                None,
                error=f"Действие изменилось или больше не разрешено: {exc.reason}",
            )
        if fresh.digest != action.digest:
            return CleanupResult(
                action.operation,
                action.display(),
                None,
                error="Действие изменилось после подтверждения — запуск отменён",
            )
        if fresh.requires_sudo and not sudo_approved:
            return CleanupResult(
                action.operation,
                action.display(),
                None,
                error="Для операции не подтверждены права root",
            )
        if fresh.prerequisite and fresh.prerequisite not in (completed_inspections or set()):
            return CleanupResult(
                action.operation,
                action.display(),
                None,
                error="Сначала выполните и подтвердите обязательный анализ",
            )

        code: int | None = 0
        error: str | None = None
        timed_out = False
        sudo = fresh.requires_sudo

        if fresh.operation == "inspect_tmp":
            reports = [inspect_directory(path) for path in ("/tmp", "/var/tmp")]
            stdout.extend(json.dumps(report.as_dict(), ensure_ascii=False) for report in reports)
            if all(report.status == "failed" for report in reports):
                code = 2
                error = "Один или несколько каталогов не удалось просканировать"
            elif any(report.status != "success" for report in reports):
                code = 1
        elif fresh.operation == "inspect_pacman_cache":
            commands = [
                ("du", ("-sh", "--", "/var/cache/pacman/pkg")),
                ("paccache", ("-d", "-k", "3")),
            ]
            for name, args in commands:
                code, error, timed_out = self._attempt(
                    name, args, sudo=sudo, stdout=stdout, stderr=stderr
                )
                if error:
                    break
            if not error:
                error = self._check_open_files(
                    "/var/cache/pacman/pkg", sudo=sudo, stdout=stdout, stderr=stderr
                )
                if error:
                    code = 1
        elif fresh.operation == "inspect_user_cache":
            report = inspect_directory(Path.home() / ".cache")
            stdout.append(json.dumps(report.as_dict(), ensure_ascii=False))
            if report.status == "failed":
                code = 2
                error = report.summary
            elif report.status == "partial_success":
                code = 1
        elif fresh.operation == "inspect_journal":
            code, error, timed_out = self._attempt(
                "journalctl", ("--disk-usage",), sudo=sudo, stdout=stdout, stderr=stderr
            )
        elif fresh.operation == "clean_tmp":
            for path in ("/tmp", "/var/tmp"):
                error = self._check_open_files(path, sudo=sudo, stdout=stdout, stderr=stderr)
                if error:
                    code = 1
                    break
                code, error, timed_out = self._attempt(
                    "systemd-tmpfiles",
                    ("--clean", f"--prefix={path}"),
                    sudo=sudo,
                    stdout=stdout,
                    stderr=stderr,
                )
                if error:
                    break
        elif fresh.operation == "clean_pacman_cache":
            error = self._check_open_files(
                "/var/cache/pacman/pkg", sudo=sudo, stdout=stdout, stderr=stderr
            )
            if error:
                code = 1
            else:
                code, error, timed_out = self._attempt(
                    "paccache", ("-r", "-k", "3"), sudo=sudo, stdout=stdout, stderr=stderr
                )
        elif fresh.operation == "vacuum_journal":
            code, error, timed_out = self._attempt(
                "journalctl",
                ("--vacuum-size=500M",),
                sudo=sudo,
                stdout=stdout,
                stderr=stderr,
            )

        after = self._free_space(action.operation)
        return CleanupResult(
            operation=fresh.operation,
            command=fresh.display(),
            exit_code=code,
            stdout="\n".join(part for part in stdout if part),
            stderr="\n".join(part for part in stderr if part),
            duration_ms=int((time.monotonic() - started) * 1000),
            timed_out=timed_out,
            error=error,
            free_before=before,
            free_after=after,
        )


def operation_spec(operation: str) -> CleanupSpec:
    try:
        return _SPECS_BY_OPERATION[operation]
    except KeyError:
        raise ValidationError("Неизвестная cleanup-операция") from None
