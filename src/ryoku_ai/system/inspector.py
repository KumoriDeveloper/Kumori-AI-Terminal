"""Общая диагностика системы: CPU, RAM, swap, диски, температуры, GPU."""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

from ._run import format_bytes, read_text, run_tool
from .storage import DiskInfo, collect_disks, collect_storage


@dataclass(slots=True)
class CpuInfo:
    model: str
    cores: int
    usage_percent: float
    load1: float
    load5: float
    load15: float


@dataclass(slots=True)
class MemoryInfo:
    total_kb: int
    available_kb: int
    swap_total_kb: int
    swap_free_kb: int

    @property
    def used_percent(self) -> float:
        if not self.total_kb:
            return 0.0
        return round((self.total_kb - self.available_kb) / self.total_kb * 100, 1)

    @property
    def total_text(self) -> str:
        return format_bytes(self.total_kb * 1024)

    @property
    def available_text(self) -> str:
        return format_bytes(self.available_kb * 1024)

    @property
    def swap_used_text(self) -> str:
        used = max(0, self.swap_total_kb - self.swap_free_kb)
        return format_bytes(used * 1024)

    @property
    def swap_total_text(self) -> str:
        return format_bytes(self.swap_total_kb * 1024)


@dataclass(slots=True)
class TemperatureInfo:
    source: str
    label: str
    celsius: float


@dataclass(slots=True)
class GpuInfo:
    name: str
    memory_total_mb: int | None = None
    memory_used_mb: int | None = None
    utilization: int | None = None
    temperature: int | None = None


@dataclass(slots=True)
class SystemReport:
    os_name: str
    kernel: str
    arch: str
    uptime_text: str
    cpu: CpuInfo
    memory: MemoryInfo
    disks: list[DiskInfo] = field(default_factory=list)
    btrfs_usage: str = ""
    btrfs_error: str | None = None
    temperatures: list[TemperatureInfo] = field(default_factory=list)
    gpus: list[GpuInfo] = field(default_factory=list)
    process_count: int = 0
    zombie_count: int = 0
    errors: list[str] = field(default_factory=list)


def _os_name() -> str:
    values: dict[str, str] = {}
    for line in read_text("/etc/os-release").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    return values.get("PRETTY_NAME") or values.get("NAME") or "Linux"


def _cpu_model() -> str:
    for line in read_text("/proc/cpuinfo").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "неизвестно"


def _read_cpu_sample() -> tuple[int, int]:
    fields = read_text("/proc/stat").splitlines()
    if not fields or not fields[0].startswith("cpu "):
        return 0, 0
    values = [int(item) for item in fields[0].split()[1:] if item.isdigit()]
    if len(values) < 4:
        return 0, 0
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return total, idle


def _cpu_usage() -> float:
    total_before, idle_before = _read_cpu_sample()
    time.sleep(0.25)
    total_after, idle_after = _read_cpu_sample()
    total_delta = total_after - total_before
    idle_delta = idle_after - idle_before
    if total_delta <= 0:
        return 0.0
    return round(max(0.0, min(100.0, (1 - idle_delta / total_delta) * 100)), 1)


def _load() -> tuple[float, float, float]:
    parts = read_text("/proc/loadavg").split()
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except (IndexError, ValueError):
        return 0.0, 0.0, 0.0


def _memory() -> MemoryInfo:
    values: dict[str, int] = {}
    for line in read_text("/proc/meminfo").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0])
    return MemoryInfo(
        total_kb=values.get("MemTotal", 0),
        available_kb=values.get("MemAvailable", 0),
        swap_total_kb=values.get("SwapTotal", 0),
        swap_free_kb=values.get("SwapFree", 0),
    )


def _uptime_text() -> str:
    try:
        seconds = float(read_text("/proc/uptime").split()[0])
    except (IndexError, ValueError):
        return "неизвестно"
    days, remainder = divmod(int(seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes = remainder // 60
    if days:
        return f"{days} д {hours} ч {minutes} мин"
    if hours:
        return f"{hours} ч {minutes} мин"
    return f"{minutes} мин"


def _temperatures() -> list[TemperatureInfo]:
    values: list[TemperatureInfo] = []
    seen: set[tuple[str, str]] = set()
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            label = (zone / "type").read_text(encoding="utf-8").strip()
            raw = (zone / "temp").read_text(encoding="utf-8").strip()
            celsius = float(raw) / 1000
        except (OSError, ValueError):
            continue
        if -50 < celsius < 200 and (label, "") not in seen:
            seen.add((label, ""))
            values.append(TemperatureInfo(source="thermal", label=label, celsius=celsius))

    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            chip = (hwmon / "name").read_text(encoding="utf-8").strip()
        except OSError:
            chip = "hwmon"
        for sensor in sorted(hwmon.glob("temp*_input")):
            try:
                celsius = float(sensor.read_text(encoding="utf-8").strip()) / 1000
            except (OSError, ValueError):
                continue
            label_path = sensor.with_name(sensor.name.replace("_input", "_label"))
            try:
                label = label_path.read_text(encoding="utf-8").strip()
            except OSError:
                label = sensor.stem.replace("temp", "temp ")
            if not (-50 < celsius < 200):
                continue
            key = (chip, label)
            if key in seen:
                continue
            seen.add(key)
            values.append(
                TemperatureInfo(source=chip, label=label, celsius=celsius)
            )
    return values[:20]


def _gpu_number(parts: list[str], index: int) -> int | None:
    try:
        return int(float(parts[index]))
    except (IndexError, ValueError):
        return None


def _gpus() -> tuple[list[GpuInfo], str | None]:
    probe = run_tool(["nvidia-smi", "-L"], timeout=5)
    if not probe.ok:
        if probe.error and "не найдена" not in probe.error:
            return [], probe.error
        return [], None
    result = run_tool(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=6,
    )
    if not result.ok:
        return [], result.error or result.stderr.strip() or "nvidia-smi недоступен"
    gpus: list[GpuInfo] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue

        gpus.append(
            GpuInfo(
                name=parts[0],
                memory_total_mb=_gpu_number(parts, 1),
                memory_used_mb=_gpu_number(parts, 2),
                utilization=_gpu_number(parts, 3),
                temperature=_gpu_number(parts, 4),
            )
        )
    return gpus, None


def _process_counts() -> tuple[int, int]:
    total = 0
    zombies = 0
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return 0, 0
    for entry in entries:
        if not entry.name.isdigit():
            continue
        total += 1
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        closing = stat.rfind(")")
        if closing != -1 and closing + 2 < len(stat) and stat[closing + 2] == "Z":
            zombies += 1
    return total, zombies


def collect_system(*, include_gpu: bool = True) -> SystemReport:
    disks, disk_errors = collect_disks()
    storage = collect_storage() if any(disk.fstype == "btrfs" for disk in disks) else None
    load1, load5, load15 = _load()
    process_count, zombie_count = _process_counts()
    gpus: list[GpuInfo] = []
    errors = list(disk_errors)
    gpu_error: str | None = None
    if include_gpu:
        gpus, gpu_error = _gpus()
        if gpu_error:
            errors.append(f"GPU: {gpu_error}")

    report = SystemReport(
        os_name=_os_name(),
        kernel=platform.release(),
        arch=platform.machine(),
        uptime_text=_uptime_text(),
        cpu=CpuInfo(
            model=_cpu_model(),
            cores=os.cpu_count() or 0,
            usage_percent=_cpu_usage(),
            load1=load1,
            load5=load5,
            load15=load15,
        ),
        memory=_memory(),
        disks=disks,
        temperatures=_temperatures(),
        gpus=gpus,
        process_count=process_count,
        zombie_count=zombie_count,
        errors=errors,
    )
    if storage is not None:
        report.btrfs_usage = storage.btrfs_usage
        report.btrfs_error = storage.btrfs_error
    return report
