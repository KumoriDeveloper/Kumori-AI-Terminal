"""Диагностика сети: интерфейсы, адреса, соединения, DNS, интернет."""

from __future__ import annotations

import json
import re
import socket
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ._run import format_bytes, read_text, run_tool

_PROCESS_RE = re.compile(r'users:\(\("([^"]+)"')


@dataclass(slots=True)
class InterfaceInfo:
    name: str
    state: str
    addresses: list[str] = field(default_factory=list)
    rx_bytes: int = 0
    tx_bytes: int = 0
    rx_rate: float = 0.0
    tx_rate: float = 0.0


@dataclass(slots=True)
class ConnectionInfo:
    netid: str
    state: str
    local: str
    peer: str
    process: str | None = None


@dataclass(slots=True)
class InternetCheck:
    name: str
    ok: bool
    detail: str = ""


@dataclass(slots=True)
class NetworkReport:
    interfaces: list[InterfaceInfo] = field(default_factory=list)
    default_route: str = "не определён"
    dns_servers: list[str] = field(default_factory=list)
    connections: list[ConnectionInfo] = field(default_factory=list)
    state_counts: dict[str, int] = field(default_factory=dict)
    top_talkers: list[tuple[str, int]] = field(default_factory=list)
    internet_checks: list[InternetCheck] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    limited_process_info: bool = False

    @property
    def internet_ok(self) -> bool:
        return any(check.ok for check in self.internet_checks)


def _read_net_dev() -> dict[str, tuple[int, int]]:
    text = read_text("/proc/net/dev")
    result: dict[str, tuple[int, int]] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, payload = line.split(":", 1)
        fields = payload.split()
        if len(fields) < 9:
            continue
        try:
            result[name.strip()] = (int(fields[0]), int(fields[8]))
        except ValueError:
            continue
    return result


def _interfaces_from_ip() -> list[InterfaceInfo]:
    result = run_tool(
        ["ip", "-j", "addr", "show"],
        timeout=5,
        max_output=200_000,
    )
    interfaces: list[InterfaceInfo] = []
    if result.ok and result.stdout.strip():
        try:
            data = json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError):
            data = None
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = item.get("ifname")
                if not isinstance(name, str):
                    continue
                addresses: list[str] = []
                for addr in item.get("addr_info") or []:
                    if not isinstance(addr, dict):
                        continue
                    local = addr.get("local")
                    family = addr.get("family")
                    prefix = addr.get("prefixlen")
                    if isinstance(local, str):
                        suffix = f"/{prefix}" if isinstance(prefix, int) else ""
                        label = "IPv6" if family == "inet6" else "IPv4"
                        addresses.append(f"{label} {local}{suffix}")
                interfaces.append(
                    InterfaceInfo(
                        name=name,
                        state=str(item.get("operstate") or "UNKNOWN").lower(),
                        addresses=addresses,
                    )
                )
            return interfaces

    fallback = run_tool(["ip", "-br", "addr", "show"], timeout=5)
    if fallback.ok:
        for line in fallback.stdout.splitlines():
            parts = line.split()
            if not parts:
                continue
            name = parts[0]
            state = parts[1] if len(parts) > 1 else "unknown"
            addresses = [" ".join(parts[2:])] if len(parts) > 2 else []
            interfaces.append(InterfaceInfo(name=name, state=state, addresses=addresses))
    return interfaces


def _default_route() -> str:
    result = run_tool(["ip", "-j", "route", "show", "default"], timeout=5)
    if result.ok and result.stdout.strip():
        try:
            routes = json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError):
            routes = None
        if isinstance(routes, list) and routes:
            route = routes[0]
            if isinstance(route, dict):
                gateway = route.get("gateway") or "?"
                device = route.get("dev") or "?"
                metric = route.get("metric")
                metric_text = f", metric {metric}" if metric is not None else ""
                return f"через {gateway} на {device}{metric_text}"
    result = run_tool(["ip", "route", "show", "default"], timeout=5)
    if result.ok and result.stdout.strip():
        return result.stdout.strip().splitlines()[0]
    return "не определён"


def _dns_servers() -> list[str]:
    servers: list[str] = []
    for line in read_text("/etc/resolv.conf").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "nameserver" and parts[1] not in servers:
            servers.append(parts[1])
    return servers


def _parse_connections(output: str) -> list[ConnectionInfo]:
    connections: list[ConnectionInfo] = []
    for line in output.splitlines():
        if not line.strip() or line.startswith(("Netid", "State")):
            continue
        parts = line.split(None, 6)
        if len(parts) < 6:
            continue
        netid, state, _recv, _send, local, peer = parts[:6]
        process = None
        if len(parts) >= 7:
            match = _PROCESS_RE.search(parts[6])
            if match:
                process = match.group(1)
        connections.append(
            ConnectionInfo(
                netid=netid,
                state=state,
                local=local,
                peer=peer,
                process=process,
            )
        )
    return connections


def _tcp_check(host: str, port: int, timeout: float = 1.6) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _dns_check() -> tuple[bool, str]:
    result = run_tool(
        ["dig", "+short", "+time=2", "+tries=1", "one.one.one.one"],
        timeout=4,
    )
    if not result.ok and result.error:
        return False, result.error
    answer = result.stdout.strip().splitlines()
    if answer:
        return True, answer[0]
    return False, "нет ответа DNS"


def collect_network(*, check_internet: bool = True) -> NetworkReport:
    report = NetworkReport()

    before = _read_net_dev()
    start = time.monotonic()
    interfaces = _interfaces_from_ip()
    time.sleep(0.35)
    after = _read_net_dev()
    elapsed = max(time.monotonic() - start, 0.01)

    if not interfaces:
        interfaces = [
            InterfaceInfo(name=name, state="unknown")
            for name in sorted(after)
        ]
        report.errors.append("ip недоступен: показаны только счётчики /proc/net/dev")

    for interface in interfaces:
        rx_before = before.get(interface.name, (0, 0))[0]
        tx_before = before.get(interface.name, (0, 0))[1]
        rx_after, tx_after = after.get(interface.name, (rx_before, tx_before))
        interface.rx_bytes = rx_after
        interface.tx_bytes = tx_after
        interface.rx_rate = max(0.0, (rx_after - rx_before) / elapsed)
        interface.tx_rate = max(0.0, (tx_after - tx_before) / elapsed)
    report.interfaces = sorted(interfaces, key=lambda item: (item.name != "lo", item.name))

    report.default_route = _default_route()
    report.dns_servers = _dns_servers()

    ss_result = run_tool(["ss", "-tunap"], timeout=6, max_output=300_000)
    if not ss_result.ok:
        limited = run_tool(["ss", "-tun"], timeout=6, max_output=300_000)
        if limited.ok:
            report.limited_process_info = True
            report.errors.append(
                "ss без root не показывает процессы — сведения о процессах ограничены"
            )
            ss_result = limited
        else:
            report.errors.append(
                ss_result.error or "ss недоступен: соединения не получены"
            )
    if ss_result.ok:
        report.connections = _parse_connections(ss_result.stdout)
        counts = Counter(conn.state for conn in report.connections)
        report.state_counts = dict(counts.most_common())
        talkers = Counter(conn.process for conn in report.connections if conn.process)
        report.top_talkers = talkers.most_common(8)

    if check_internet:
        with ThreadPoolExecutor(max_workers=3) as pool:
            tcp1 = pool.submit(_tcp_check, "1.1.1.1", 443)
            tcp2 = pool.submit(_tcp_check, "8.8.8.8", 53)
            dns = pool.submit(_dns_check)
            for name, future in (
                ("TCP 1.1.1.1:443", tcp1),
                ("TCP 8.8.8.8:53", tcp2),
                ("DNS one.one.one.one", dns),
            ):
                try:
                    value = future.result(timeout=5)
                except Exception:
                    value = (False, "таймаут") if name.startswith("DNS") else False
                if name.startswith("DNS"):
                    ok, detail = value  # type: ignore[misc]
                    report.internet_checks.append(
                        InternetCheck(name=name, ok=bool(ok), detail=str(detail))
                    )
                else:
                    report.internet_checks.append(
                        InternetCheck(
                            name=name,
                            ok=bool(value),
                            detail="доступен" if value else "нет соединения",
                        )
                    )

    return report


def traffic_summary(report: NetworkReport) -> str:
    parts = [
        f"{interface.name}: ↓ {format_bytes(interface.rx_rate)}/s "
        f"↑ {format_bytes(interface.tx_rate)}/s"
        for interface in report.interfaces
        if interface.name != "lo"
    ]
    return "; ".join(parts) if parts else "нет данных"
