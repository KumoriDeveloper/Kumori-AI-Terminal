"""Процессы: нагрузка CPU/RAM, зомби, сетевая активность."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ._run import run_tool

_PROCESS_RE = re.compile(r'users:\(\("([^"]+)"')


@dataclass(slots=True)
class ProcessInfo:
    pid: int
    user: str
    cpu: float
    mem: float
    elapsed: str
    command: str


@dataclass(slots=True)
class ProcessesReport:
    top_cpu: list[ProcessInfo] = field(default_factory=list)
    top_mem: list[ProcessInfo] = field(default_factory=list)
    zombies: list[ProcessInfo] = field(default_factory=list)
    total: int = 0
    network_active: list[tuple[str, int]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    limited_network_info: bool = False


def _parse_ps(output: str, limit: int) -> list[ProcessInfo]:
    result: list[ProcessInfo] = []
    for line in output.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        try:
            process = ProcessInfo(
                pid=int(parts[0]),
                user=parts[1],
                cpu=float(parts[2]),
                mem=float(parts[3]),
                elapsed=parts[4],
                command=parts[5].strip(),
            )
        except ValueError:
            continue
        result.append(process)
        if len(result) >= limit:
            break
    return result


def _total_processes() -> int:
    count = 0
    try:
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                count += 1
    except OSError:
        return 0
    return count


def _zombies(limit: int = 10) -> list[ProcessInfo]:
    result = run_tool(
        ["ps", "-ww", "-eo", "pid,user,pcpu,pmem,etime,stat,args"],
        timeout=5,
        max_output=300_000,
    )
    if not result.ok:
        return []
    zombies: list[ProcessInfo] = []
    for line in result.stdout.splitlines():
        parts = line.split(None, 6)
        if len(parts) < 7 or not parts[5].startswith("Z"):
            continue
        try:
            zombies.append(
                ProcessInfo(
                    pid=int(parts[0]),
                    user=parts[1],
                    cpu=float(parts[2]),
                    mem=float(parts[3]),
                    elapsed=parts[4],
                    command=parts[6].strip(),
                )
            )
        except ValueError:
            continue
        if len(zombies) >= limit:
            break
    return zombies


def _network_processes() -> tuple[list[tuple[str, int]], bool, str | None]:
    result = run_tool(["ss", "-tunp"], timeout=6, max_output=300_000)
    limited = False
    if not result.ok:
        result = run_tool(["ss", "-tun"], timeout=6, max_output=300_000)
        limited = True
    if not result.ok:
        return [], limited, result.error or "ss недоступен"
    counter: Counter[str] = Counter()
    for line in result.stdout.splitlines():
        match = _PROCESS_RE.search(line)
        if match:
            counter[match.group(1)] += 1
    return counter.most_common(10), limited, None


def collect_processes(limit: int = 10) -> ProcessesReport:
    report = ProcessesReport(total=_total_processes())
    cpu = run_tool(
        ["ps", "-ww", "-eo", "pid,user,pcpu,pmem,etime,args", "--sort=-pcpu"],
        timeout=6,
        max_output=300_000,
    )
    if cpu.ok:
        report.top_cpu = _parse_ps(cpu.stdout, limit)
    else:
        report.errors.append(cpu.error or "ps недоступен")

    mem = run_tool(
        ["ps", "-ww", "-eo", "pid,user,pcpu,pmem,etime,args", "--sort=-pmem"],
        timeout=6,
        max_output=300_000,
    )
    if mem.ok:
        report.top_mem = _parse_ps(mem.stdout, limit)
    elif not report.errors:
        report.errors.append(mem.error or "ps недоступен")

    report.zombies = _zombies()
    report.network_active, report.limited_network_info, error = _network_processes()
    if error:
        report.errors.append(error)
    elif report.limited_network_info:
        report.errors.append(
            "Нет прав на просмотр процессов в ss — показаны только соединения"
        )
    return report
