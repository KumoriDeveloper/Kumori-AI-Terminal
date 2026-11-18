"""systemd, журнал ошибок, Docker и Steam."""

from __future__ import annotations

from dataclasses import dataclass, field

from ._run import run_tool, which


@dataclass(slots=True)
class FailedUnit:
    name: str
    load: str
    active: str
    sub: str
    description: str


@dataclass(slots=True)
class DockerInfo:
    installed: bool = False
    service_state: str = "не проверено"
    version: str | None = None
    containers: list[str] = field(default_factory=list)
    disk_usage: str = ""
    error: str | None = None


@dataclass(slots=True)
class ServiceReport:
    system_state: str = "неизвестно"
    failed_units: list[FailedUnit] = field(default_factory=list)
    running_services: int = 0
    journal_errors: list[str] = field(default_factory=list)
    docker: DockerInfo = field(default_factory=DockerInfo)
    steam_running: bool = False
    steam_processes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _system_state() -> str:
    result = run_tool(["systemctl", "is-system-running"], timeout=5)
    value = result.stdout.strip() or result.stderr.strip()
    if value:
        return value
    return result.error or "неизвестно"


def _failed_units() -> list[FailedUnit]:
    result = run_tool(
        ["systemctl", "--failed", "--no-pager", "--no-legend", "--plain"],
        timeout=6,
    )
    if not result.ok and not result.stdout.strip():
        return []
    units: list[FailedUnit] = []
    for line in result.stdout.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5 or not parts[0].endswith((".service", ".timer", ".socket", ".mount")):
            continue
        units.append(
            FailedUnit(
                name=parts[0],
                load=parts[1],
                active=parts[2],
                sub=parts[3],
                description=parts[4],
            )
        )
    return units


def _running_services() -> int:
    result = run_tool(
        [
            "systemctl",
            "list-units",
            "--type=service",
            "--state=running",
            "--no-pager",
            "--no-legend",
            "--plain",
        ],
        timeout=6,
    )
    if not result.ok:
        return 0
    return sum(1 for line in result.stdout.splitlines() if line.strip().endswith(".service"))


def _journal_errors(limit: int = 15) -> tuple[list[str], str | None]:
    result = run_tool(
        ["journalctl", "-p", "3", "-b", "-n", str(limit), "--no-pager", "--no-legend"],
        timeout=8,
    )
    if not result.ok:
        reason = result.stderr.strip() or result.error or "journalctl недоступен"
        return [], reason
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[-limit:], None


def _docker() -> DockerInfo:
    info = DockerInfo(installed=which("docker") is not None)
    if not info.installed:
        return info
    active = run_tool(["systemctl", "is-active", "docker"], timeout=5)
    info.service_state = active.stdout.strip() or active.stderr.strip() or "unknown"

    version = run_tool(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        timeout=8,
        max_output=20_000,
    )
    if version.ok and version.stdout.strip():
        info.version = version.stdout.strip().splitlines()[0]
    else:
        info.error = version.stderr.strip() or version.error or "docker недоступен"

    containers = run_tool(
        ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}"],
        timeout=10,
        max_output=100_000,
    )
    if containers.ok:
        info.containers = [
            line.strip() for line in containers.stdout.splitlines() if line.strip()
        ][:30]
    elif not info.error:
        info.error = containers.stderr.strip() or containers.error

    disk = run_tool(["docker", "system", "df"], timeout=10, max_output=50_000)
    if disk.ok:
        info.disk_usage = disk.stdout.strip()
    return info


def _steam() -> tuple[bool, list[str], str | None]:
    result = run_tool(["pgrep", "-a", "steam"], timeout=5, max_output=50_000)
    if result.ok:
        processes = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ][:10]
        return bool(processes), processes, None
    if result.exit_code == 1 and not result.stdout.strip():
        return False, [], None
    return False, [], result.error


def collect_services() -> ServiceReport:
    report = ServiceReport()
    report.system_state = _system_state()
    if report.system_state not in {"running", "degraded", "starting", "initializing", "maintenance"}:
        report.errors.append(f"systemctl: {report.system_state}")
    report.failed_units = _failed_units()
    report.running_services = _running_services()
    report.journal_errors, journal_error = _journal_errors()
    if journal_error:
        report.errors.append(f"journalctl: {journal_error}")
    report.docker = _docker()
    if report.docker.error:
        report.errors.append(f"docker: {report.docker.error}")
    report.steam_running, report.steam_processes, steam_error = _steam()
    if steam_error:
        report.errors.append(f"steam: {steam_error}")
    return report
