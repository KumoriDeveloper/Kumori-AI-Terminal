"""Безопасный запуск внутренних диагностических утилит."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..execution.executor import safe_env

SAFE_PATH = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


@dataclass(slots=True)
class ToolResult:
    argv: tuple[str, ...]
    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    error: str | None = None


def which(name: str) -> str | None:
    return shutil.which(name, path=SAFE_PATH)


def run_tool(
    argv: list[str],
    *,
    timeout: float = 6.0,
    max_output: int = 200_000,
    cwd: Path | None = None,
) -> ToolResult:
    if not argv:
        return ToolResult((), False, error="Пустая команда")
    executable = which(argv[0])
    if not executable:
        return ToolResult(tuple(argv), False, error=f"Утилита не найдена: {argv[0]}")
    try:
        completed = subprocess.run(  # noqa: S603
            [executable, *argv[1:]],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            cwd=str(cwd) if cwd else None,
            env=safe_env(),
            shell=False,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(tuple(argv), False, error="Превышено время выполнения")
    except (OSError, ValueError) as exc:
        return ToolResult(tuple(argv), False, error=str(exc))

    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    truncated_note = ""
    if len(stdout) > max_output:
        stdout = stdout[:max_output] + "\n… вывод обрезан"
        truncated_note = "вывод обрезан"
    if len(stderr) > max_output:
        stderr = stderr[:max_output] + "\n… вывод обрезан"
    return ToolResult(
        tuple(argv),
        completed.returncode == 0,
        stdout=stdout,
        stderr=stderr,
        exit_code=completed.returncode,
        error=truncated_note or None,
    )


def read_text(path: str, default: str = "") -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return default


def format_bytes(value: float | int) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(number) < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(number)} B"
            return f"{number:.1f} {unit}"
        number /= 1024
    return f"{number:.1f} TiB"
