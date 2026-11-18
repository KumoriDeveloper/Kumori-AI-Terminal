"""Единственное место, где Ryoku AI Terminal запускает процессы.

Никакого shell, eval, sh -c или bash -c. Только argv, абсолютный путь к
исполняемому файлу, минимальное окружение, таймаут и ограничение вывода.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from ..config import Config
from ..execution.models import ValidatedCommand, ValidationError
from ..execution.validator import CommandValidator

ExecutionStatus = Literal[
    "success",
    "partial_success",
    "failed",
    "cancelled",
    "timeout",
]
_PERMISSION_ERROR = re.compile(r"permission denied|operation not permitted|access denied", re.I)
_QUOTED_PATH = re.compile(r"['\"]([^'\"]+)['\"]")
_PARTIAL_ERROR = re.compile(
    r"permission denied|operation not permitted|access denied|no such file|cannot stat|vanished|disappeared",
    re.I,
)


@dataclass(slots=True)
class ExecutionResult:
    argv: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    truncated: bool = False
    error: str | None = None
    started_at: str = ""
    status: ExecutionStatus = field(init=False)
    warnings: list[str] = field(default_factory=list)
    summary: str = ""
    inaccessible_count: int = 0

    def __post_init__(self) -> None:
        permission_lines = [
            line for line in self.stderr.splitlines() if _PERMISSION_ERROR.search(line)
        ]
        incomplete_lines = [
            line for line in self.stderr.splitlines() if _PARTIAL_ERROR.search(line)
        ]
        inaccessible = {
            match.group(1)
            for line in permission_lines
            if (match := _QUOTED_PATH.search(line)) is not None
        }
        self.inaccessible_count = len(inaccessible) or len(permission_lines)
        if self.inaccessible_count:
            label = "каталогов" if Path(self.argv[0]).name == "du" else "путей"
            noun = "Недоступных каталогов" if label == "каталогов" else "Недоступных путей"
            self.warnings.append(
                f"{noun}: {self.inaccessible_count} (Permission denied)."
            )

        if self.timed_out:
            self.status = "timeout"
        elif self.stdout.strip() and incomplete_lines and self.error is None:
            self.status = "partial_success"
        elif self.exit_code == 0 and self.error is None:
            self.status = "success"
        elif self.stdout.strip() and self.exit_code not in (0, None) and (
            incomplete_lines or self.error
        ):
            self.status = "partial_success"
        else:
            self.status = "failed"

        if not self.summary:
            if self.status == "success":
                self.summary = "Команда выполнена успешно."
            elif self.status == "partial_success":
                if Path(self.argv[0]).name == "du":
                    self.summary = (
                        "Полезный вывод получен; результат неполный, размер является "
                        "приблизительной нижней оценкой."
                    )
                else:
                    self.summary = "Полезный диагностический вывод получен, но результат неполный."
            elif self.status == "timeout":
                self.summary = "Время выполнения команды истекло."
            elif self.status == "cancelled":
                self.summary = "Выполнение отменено."
            else:
                self.summary = self.error or "Команда не смогла получить результат."
        if self.status == "partial_success" and not self.warnings:
            self.warnings.append(
                f"Команда завершилась с кодом {self.exit_code}; часть результата может отсутствовать."
            )

    @classmethod
    def cancelled(cls, argv: tuple[str, ...], *, summary: str = "Выполнение отменено.") -> ExecutionResult:
        result = cls(argv, None, "", "", 0, summary=summary)
        result.status = "cancelled"
        return result

    def structured(self) -> dict[str, object]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "warnings": list(self.warnings),
            "summary": self.summary,
            "inaccessible_count": self.inaccessible_count,
        }

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timed_out and self.exit_code == 0

    @property
    def combined(self) -> str:
        parts: list[str] = []
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.stderr.strip():
            parts.append("[stderr]\n" + self.stderr.rstrip())
        if self.error:
            parts.append(f"[error] {self.error}")
        return "\n".join(parts).strip()


def safe_env() -> dict[str, str]:
    allowed = (
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "XDG_RUNTIME_DIR",
        "DBUS_SESSION_BUS_ADDRESS",
        "XDG_SESSION_TYPE",
        "WAYLAND_DISPLAY",
    )
    env = {name: os.environ[name] for name in allowed if name in os.environ}
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    return env


class _LimitedReader(threading.Thread):
    def __init__(self, pipe, byte_limit: int):
        super().__init__(daemon=True)
        self.pipe = pipe
        self.byte_limit = byte_limit
        self.buffer = bytearray()
        self.truncated = False
        self.error: str | None = None

    def run(self) -> None:
        try:
            while True:
                chunk = self.pipe.read(8192)
                if not chunk:
                    break
                remaining = self.byte_limit - len(self.buffer)
                if remaining > 0:
                    self.buffer.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self.truncated = True
        except (OSError, ValueError) as exc:
            self.error = str(exc)
        finally:
            with contextlib.suppress(OSError):
                self.pipe.close()

    def text(self) -> str:
        return bytes(self.buffer).decode("utf-8", errors="replace")


class Executor:
    def __init__(
        self,
        config: Config,
        *,
        validator: CommandValidator | None = None,
        revalidator: Callable[[ValidatedCommand], ValidatedCommand] | None = None,
        cwd: Path | None = None,
    ):
        self.config = config
        self.validator = validator or CommandValidator(
            allow_sudo=config.allow_sudo,
            cwd=cwd,
        )
        self._revalidate = revalidator or self.validator.revalidate
        self.cwd = (cwd or Path.cwd()).resolve()

    def _sudo_path(self) -> str | None:
        resolver = self.validator.resolver
        return resolver("sudo")

    def execute(
        self,
        command: ValidatedCommand,
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
        max_output_chars: int | None = None,
    ) -> ExecutionResult:
        started = time.monotonic()
        started_at = datetime.now(UTC).isoformat(timespec="seconds")
        timeout_value = min(
            max(1.0, float(timeout if timeout is not None else self.config.command_timeout)),
            120.0,
        )
        output_limit = max(
            1000,
            min(
                int(
                    max_output_chars
                    if max_output_chars is not None
                    else self.config.max_output_chars
                ),
                200_000,
            ),
        )

        try:
            fresh = self._revalidate(command)
        except ValidationError as exc:
            return ExecutionResult(
                argv=command.argv,
                exit_code=None,
                stdout="",
                stderr="",
                duration_ms=int((time.monotonic() - started) * 1000),
                error=f"Проверка перед запуском не пройдена: {exc.reason}",
                started_at=started_at,
            )

        if fresh.digest != command.digest:
            return ExecutionResult(
                argv=command.argv,
                exit_code=None,
                stdout="",
                stderr="",
                duration_ms=int((time.monotonic() - started) * 1000),
                error="Команда изменилась после подтверждения — запуск отменён",
                started_at=started_at,
            )

        argv: list[str] = []
        if fresh.requires_sudo:
            sudo_path = self._sudo_path()
            if not sudo_path:
                return ExecutionResult(
                    argv=fresh.argv,
                    exit_code=None,
                    stdout="",
                    stderr="",
                    duration_ms=int((time.monotonic() - started) * 1000),
                    error="sudo не найден; выполните команду вручную в терминале",
                    started_at=started_at,
                )
            argv = [sudo_path, "-n", "--", fresh.executable, *fresh.argv[1:]]
        else:
            argv = [fresh.executable, *fresh.argv[1:]]

        workdir = (cwd or self.cwd).expanduser()
        if not workdir.is_dir():
            return ExecutionResult(
                argv=fresh.argv,
                exit_code=None,
                stdout="",
                stderr="",
                duration_ms=int((time.monotonic() - started) * 1000),
                error=f"Рабочий каталог не существует: {workdir}",
                started_at=started_at,
            )

        try:
            process = subprocess.Popen(  # noqa: S603
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(workdir),
                env=safe_env(),
                shell=False,
                start_new_session=True,
                close_fds=True,
            )
        except (OSError, ValueError) as exc:
            return ExecutionResult(
                argv=fresh.argv,
                exit_code=None,
                stdout="",
                stderr="",
                duration_ms=int((time.monotonic() - started) * 1000),
                error=f"Не удалось запустить команду: {exc}",
                started_at=started_at,
            )

        assert process.stdout is not None
        assert process.stderr is not None
        byte_limit = output_limit * 4
        stdout_reader = _LimitedReader(process.stdout, byte_limit)
        stderr_reader = _LimitedReader(process.stderr, byte_limit)
        stdout_reader.start()
        stderr_reader.start()

        timed_out = False
        try:
            process.wait(timeout=timeout_value)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                with contextlib.suppress(OSError):
                    process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=2)

        stdout_reader.join(timeout=3)
        stderr_reader.join(timeout=3)
        if stdout_reader.is_alive() or stderr_reader.is_alive():
            for pipe in (process.stdout, process.stderr):
                with contextlib.suppress(OSError):
                    pipe.close()
            stdout_reader.join(timeout=1)
            stderr_reader.join(timeout=1)

        stdout_text = stdout_reader.text()
        stderr_text = stderr_reader.text()
        truncated = stdout_reader.truncated or stderr_reader.truncated
        if len(stdout_text) > output_limit:
            stdout_text = stdout_text[:output_limit]
            truncated = True
        if len(stderr_text) > output_limit:
            stderr_text = stderr_text[:output_limit]
            truncated = True

        error: str | None = None
        if stdout_reader.error:
            error = f"Ошибка чтения stdout: {stdout_reader.error}"
        elif stderr_reader.error:
            error = f"Ошибка чтения stderr: {stderr_reader.error}"

        return ExecutionResult(
            argv=fresh.argv,
            exit_code=None if timed_out else process.returncode,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_ms=int((time.monotonic() - started) * 1000),
            timed_out=timed_out,
            truncated=truncated,
            error=error,
            started_at=started_at,
        )
