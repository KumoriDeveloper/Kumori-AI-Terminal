"""Минимальный локальный контекст для AI-запросов.

Контекст собирается только из открытых системных файлов и переменных
окружения. Содержимое домашних файлов, история команд, пароли и ключи сюда
не попадают.
"""

from __future__ import annotations

import os
import platform
import shutil
from datetime import datetime
from pathlib import Path


def _read(path: str, default: str = "") -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return default


def _distro() -> str:
    values: dict[str, str] = {}
    for line in _read("/etc/os-release").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    return values.get("PRETTY_NAME") or values.get("NAME") or "Linux"


def _cpu_model() -> str:
    for line in _read("/proc/cpuinfo").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "неизвестно"


def _memory() -> str:
    values: dict[str, int] = {}
    for line in _read("/proc/meminfo").splitlines():
        if ":" in line:
            key, raw = line.split(":", 1)
            parts = raw.strip().split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0])
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    if not total:
        return "неизвестно"
    used_percent = round((total - available) / total * 100) if total else 0
    return f"{total / 1024 / 1024:.1f} GiB всего, занято ~{used_percent}%"


def _root_disk(cwd: Path) -> str:
    try:
        usage = shutil.disk_usage(cwd)
    except OSError:
        return "неизвестно"
    total = usage.total / 1024**3
    free = usage.free / 1024**3
    return f"{free:.1f} GiB свободно из {total:.1f} GiB"


def _detect_desktop() -> str:
    desktop = os.environ.get("XDG_CURRENT_DESKTOP") or os.environ.get("DESKTOP_SESSION")
    session_type = os.environ.get("XDG_SESSION_TYPE")
    wayland = os.environ.get("WAYLAND_DISPLAY")
    parts: list[str] = []
    if desktop:
        parts.append(desktop)
    if session_type:
        parts.append(session_type)
    if wayland:
        parts.append("Wayland")
    return " / ".join(dict.fromkeys(parts)) or "неизвестно"


def collect_context(cwd: Path | None = None) -> str:
    """Собрать короткий текстовый контекст (без секретов)."""

    cwd = (cwd or Path.cwd()).resolve()
    shell = Path(os.environ.get("SHELL", "fish")).name
    load = _read("/proc/loadavg").split()[:3]
    load_text = " ".join(load) if load else "неизвестно"
    date = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"Дата: {date}",
        f"ОС: {_distro()} ({platform.machine()})",
        f"Ядро: {platform.release()}",
        f"Рабочий стол: {_detect_desktop()}",
        f"Оболочка: {shell}",
        f"Каталог: {cwd}",
        f"CPU: {_cpu_model()}",
        f"Память: {_memory()}",
        f"Корневой диск: {_root_disk(cwd)}",
        f"Load average: {load_text}",
        f"Python: {platform.python_version()}",
    ]

    text = "\n".join(lines)
    return text[:1200]
