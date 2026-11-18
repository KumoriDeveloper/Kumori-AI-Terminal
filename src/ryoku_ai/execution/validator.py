"""Строгая проверка команд перед показом пользователю.

Правила:
* команда задаётся argv, а не shell-строкой;
* бинарник обязан быть в белом списке и найден в системном каталоге;
* shell-метасимволы и глобы запрещены;
* аргументы проверяются отдельно для каждой утилиты;
* sudo не входит в argv и требует отдельного подтверждения;
* секретные пути читать нельзя.
"""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .models import ProposedCommand, RiskLevel, ValidatedCommand, ValidationError, compute_digest
from .policy import (
    ALLOWED_BINARIES,
    DENIED_BINARIES,
    GLOB_CHARACTERS,
    SAFE_EXEC_DIRS,
    SAFE_PATH_ROOTS,
    SAFE_READ_FILES,
    SENSITIVE_PATH_FRAGMENTS,
    SHELL_METACHARACTERS,
    SUDO_ALLOWED,
    policy_note,
)

_BINARY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_IFACE_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,32}$")
_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@:\\-]{1,160}$")
_HOST_RE = re.compile(r"^[A-Za-z0-9._:%/-]{1,253}$")
_DOMAIN_RE = re.compile(r"^[A-Za-z0-9._-]{1,253}$")
_PACKAGE_RE = re.compile(r"^[A-Za-z0-9@._+:-]{1,120}$")
_CONTAINER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$")
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:@/+,-]{1,160}$")
_NUMBER_RE = re.compile(r"^\d+$")
_FLOAT_RE = re.compile(r"^\d+(?:\.\d+)?$")
_TIME_RE = re.compile(r"^[0-9A-Za-z:._ +@-]{1,60}$")

_MAX_ARGS = 32
_MAX_ARG_LEN = 1000
_MAX_TOTAL_LEN = 4096


@dataclass(frozen=True)
class FlagSpec:
    short: frozenset[str] = field(default_factory=frozenset)
    short_value: frozenset[str] = field(default_factory=frozenset)
    long: frozenset[str] = field(default_factory=frozenset)
    long_value: frozenset[str] = field(default_factory=frozenset)
    multi_short: frozenset[str] = field(default_factory=frozenset)


@dataclass
class ParsedArgs:
    positionals: list[str] = field(default_factory=list)
    flags: list[tuple[str, str | None]] = field(default_factory=list)

    def has(self, *names: str) -> bool:
        return any(name in names for name, _ in self.flags)

    def value(self, *names: str) -> str | None:
        for name, value in reversed(self.flags):
            if name in names:
                return value
        return None

    def values(self, *names: str) -> list[str]:
        return [value for name, value in self.flags if name in names and value is not None]


def _parse_flags(args: list[str], spec: FlagSpec) -> ParsedArgs:
    parsed = ParsedArgs()
    index = 0
    end_of_flags = False
    while index < len(args):
        arg = args[index]
        if end_of_flags:
            parsed.positionals.append(arg)
            index += 1
            continue
        if arg == "--":
            end_of_flags = True
            index += 1
            continue
        if arg.startswith("--"):
            name, has_equal, value = arg.partition("=")
            if name in spec.long_value:
                if not has_equal:
                    index += 1
                    if index >= len(args):
                        raise ValidationError(f"Флаг {name} требует значение")
                    value = args[index]
                parsed.flags.append((name, value))
            elif name in spec.long:
                if has_equal:
                    raise ValidationError(f"Флаг {name} не принимает значение")
                parsed.flags.append((name, None))
            else:
                raise ValidationError(f"Флаг {name} запрещён для этой команды")
            index += 1
            continue
        if arg.startswith("-") and arg != "-":
            if arg in spec.multi_short:
                parsed.flags.append((arg, None))
                index += 1
                continue
            chars = arg[1:]
            pos = 0
            while pos < len(chars):
                char = chars[pos]
                flag = f"-{char}"
                if char in spec.short_value:
                    rest = chars[pos + 1 :]
                    if rest:
                        value = rest
                    else:
                        index += 1
                        if index >= len(args):
                            raise ValidationError(f"Флаг {flag} требует значение")
                        value = args[index]
                    parsed.flags.append((flag, value))
                    pos = len(chars)
                    break
                if char not in spec.short:
                    raise ValidationError(f"Флаг {flag} запрещён для этой команды")
                parsed.flags.append((flag, None))
                pos += 1
            index += 1
            continue
        parsed.positionals.append(arg)
        index += 1
    return parsed


def _int_value(value: str | None, *, name: str, minimum: int, maximum: int) -> int:
    if value is None or not _NUMBER_RE.match(value):
        raise ValidationError(f"{name}: ожидается целое число")
    number = int(value)
    if number < minimum or number > maximum:
        raise ValidationError(f"{name}: допустимый диапазон {minimum}..{maximum}")
    return number


def _float_value(value: str | None, *, name: str, minimum: float, maximum: float) -> float:
    if value is None or not _FLOAT_RE.match(value):
        raise ValidationError(f"{name}: ожидается число")
    number = float(value)
    if number < minimum or number > maximum:
        raise ValidationError(f"{name}: допустимый диапазон {minimum}..{maximum}")
    return number


def _is_sensitive(path_text: str) -> bool:
    lowered = path_text.lower().replace("\\", "/")
    return any(fragment in lowered for fragment in SENSITIVE_PATH_FRAGMENTS)


def _resolved_path(raw: str, cwd: Path) -> tuple[Path, str]:
    if not raw or "\x00" in raw:
        raise ValidationError("Пустой или недопустимый путь")
    if _is_sensitive(raw):
        raise ValidationError("Путь похож на секретный — чтение запрещено")
    expanded = Path(raw).expanduser()
    candidate = expanded if expanded.is_absolute() else cwd / expanded
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        raise ValidationError(f"Не удалось проверить путь: {raw}") from None
    if _is_sensitive(str(resolved)):
        raise ValidationError("Итоговый путь указывает на секретный файл")
    return resolved, str(expanded)


def _path_in_roots(raw: str, cwd: Path, roots: tuple[str, ...]) -> Path:
    resolved, _ = _resolved_path(raw, cwd)
    allowed = [Path(root).resolve(strict=False) for root in roots]
    allowed.append(cwd.resolve(strict=False))
    for root in allowed:
        try:
            if resolved == root or resolved.is_relative_to(root):
                return resolved
        except ValueError:
            continue
    raise ValidationError("Путь вне разрешённых каталогов")


def _safe_read_file(raw: str, cwd: Path) -> Path:
    resolved, expanded = _resolved_path(raw, cwd)
    absolute = str(resolved)
    if expanded in SAFE_READ_FILES or absolute in SAFE_READ_FILES:
        return resolved
    raise ValidationError("Чтение этого файла не разрешено (только безопасные системные файлы)")


def _safe_any_path(raw: str, cwd: Path) -> Path:
    resolved, _ = _resolved_path(raw, cwd)
    return resolved


def _require_between(values: list[str], minimum: int, maximum: int, label: str) -> None:
    if not (minimum <= len(values) <= maximum):
        raise ValidationError(f"{label}: ожидается от {minimum} до {maximum} аргументов")


def _check_units(positionals: list[str], limit: int = 3) -> None:
    for unit in positionals:
        if not _UNIT_RE.match(unit):
            raise ValidationError(f"Недопустимое имя unit: {unit}")
    if len(positionals) > limit:
        raise ValidationError("Слишком много unit-аргументов")


# --- Сетевые утилиты -------------------------------------------------------


def _v_ss(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("tunapleor46HmsigxwSZ"),
        short_value=frozenset("f"),
        long=frozenset(
            {
                "--tcp",
                "--udp",
                "--all",
                "--numeric",
                "--processes",
                "--listening",
                "--summary",
                "--info",
                "--options",
                "--extended",
                "--memory",
                "--timers",
                "--unix",
                "--packet",
                "--raw",
                "--dccp",
                "--mptcp",
                "--ipv4",
                "--ipv6",
                "--no-header",
                "--resolve",
                "--one-line",
                "--bound-inactive",
                "--bpf",
                "--context",
                "--query",
                "--help",
                "--version",
            }
        ),
        long_value=frozenset({"--family"}),
    )
    parsed = _parse_flags(args, spec)
    if parsed.positionals:
        raise ValidationError("ss: фильтры-выражения не разрешены, используйте флаги")
    family = parsed.value("-f", "--family")
    if family and family not in {"inet", "inet6", "unix", "packet", "netlink", "vsock", "xdp"}:
        raise ValidationError("ss: недопустимое семейство адресов")


def _v_ip(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        multi_short=frozenset({"-br", "-j", "-s", "-d", "-4", "-6", "-c", "-o", "-a"}),
        short=frozenset(),
        short_value=frozenset("f"),
        long=frozenset(
            {
                "--brief",
                "--json",
                "--stats",
                "--details",
                "--oneline",
                "--color",
                "--human",
                "--all",
                "--numeric",
                "--resolve",
                "--help",
                "--version",
            }
        ),
        long_value=frozenset({"--family"}),
    )
    parsed = _parse_flags(args, spec)
    family = parsed.value("-f", "--family")
    if family and family not in {"inet", "inet6", "link", "bridge", "mpls"}:
        raise ValidationError("ip: недопустимое семейство")
    if not parsed.positionals:
        raise ValidationError("ip: укажите объект (addr, link, route, neigh)")
    objects = {
        "address",
        "addr",
        "a",
        "link",
        "l",
        "route",
        "r",
        "neigh",
        "neighbor",
        "n",
        "rule",
        "ru",
        "maddress",
        "maddr",
    }
    modify = {
        "add",
        "del",
        "delete",
        "set",
        "change",
        "replace",
        "flush",
        "append",
        "prepend",
        "exec",
        "monitor",
        "save",
        "restore",
        "batch",
        "netns",
        "update",
    }
    if parsed.positionals[0] not in objects:
        raise ValidationError("ip: разрешены только объекты addr/link/route/neigh/rule")
    if len(parsed.positionals) > 5:
        raise ValidationError("ip: слишком много аргументов")
    for token in parsed.positionals[1:]:
        if token in modify:
            raise ValidationError(f"ip: изменяющая операция «{token}» запрещена")
        if token in {"show", "list", "ls", "get", "help", "dev", "default", "main", "local", "all", "table"}:
            continue
        if _IFACE_RE.match(token) or _SAFE_TOKEN_RE.match(token):
            continue
        raise ValidationError(f"ip: недопустимый аргумент «{token}»")


def _v_ping(args: list[str], ctx: _Context) -> str:
    spec = FlagSpec(
        short=frozenset("46nq"),
        short_value=frozenset("cWwsi"),
        long=frozenset({"--numeric", "--quiet", "--ipv4", "--ipv6"}),
        long_value=frozenset(
            {"--count", "--deadline", "--timeout", "--size", "--interval"}
        ),
    )
    parsed = _parse_flags(args, spec)
    _require_between(parsed.positionals, 1, 1, "ping")
    host = parsed.positionals[0]
    if not _HOST_RE.match(host):
        raise ValidationError("ping: недопустимый адрес или имя хоста")
    count_raw = parsed.value("-c", "--count")
    count = 4 if count_raw is None else _int_value(count_raw, name="-c", minimum=1, maximum=10)
    if count_raw is None:
        args[0:0] = ["-c", "4"]
    deadline = parsed.value("-w", "--deadline")
    if deadline is not None:
        _float_value(deadline, name="-w", minimum=1.0, maximum=15.0)
    timeout = parsed.value("-W", "--timeout")
    if timeout is not None:
        _float_value(timeout, name="-W", minimum=1.0, maximum=10.0)
    size = parsed.value("-s", "--size")
    if size is not None:
        _int_value(size, name="-s", minimum=1, maximum=1400)
    interval = parsed.value("-i", "--interval")
    if interval is not None:
        _float_value(interval, name="-i", minimum=0.2, maximum=5.0)
    if count > 10:
        raise ValidationError("ping: не более 10 пакетов")
    return "Число пакетов ограничено; при отсутствии -c добавлено -c 4."


def _v_dig(args: list[str], ctx: _Context) -> None:
    plus_allowed = {
        "+short",
        "+noall",
        "+answer",
        "+authority",
        "+additional",
        "+stats",
        "+comments",
        "+question",
        "+ttlid",
        "+recurse",
        "+tcp",
        "+notcp",
        "+dnssec",
        "+cdflag",
    }
    plus_value = {"+time": (1, 10), "+tries": (1, 3)}
    positionals: list[str] = []
    for arg in args:
        if arg.startswith("-"):
            raise ValidationError("dig: короткие флаги запрещены, используйте +опции")
        if arg.startswith("+"):
            name, has_equal, value = arg.partition("=")
            if name in plus_allowed and not has_equal:
                continue
            if name in plus_value:
                if not has_equal:
                    raise ValidationError(f"dig: опция {name} требует значение")
                low, high = plus_value[name]
                _int_value(value, name=name, minimum=low, maximum=high)
                continue
            raise ValidationError(f"dig: опция {name} запрещена")
        positionals.append(arg)
    if not positionals:
        raise ValidationError("dig: укажите имя хоста")
    servers = [item for item in positionals if item.startswith("@")]
    rest = [item for item in positionals if not item.startswith("@")]
    if len(servers) > 1 or not rest:
        raise ValidationError("dig: недопустимый набор аргументов")
    for server in servers:
        if not re.match(r"^@[A-Za-z0-9.:-]{1,253}$", server):
            raise ValidationError("dig: недопустимый DNS-сервер")
    host = rest[0]
    if not _HOST_RE.match(host):
        raise ValidationError("dig: недопустимое имя хоста")
    if len(rest) > 2:
        raise ValidationError("dig: слишком много аргументов")
    if len(rest) == 2:
        qtype = rest[1].upper()
        if qtype not in {"A", "AAAA", "MX", "NS", "TXT", "CNAME", "SOA", "SRV", "CAA", "PTR"}:
            raise ValidationError("dig: недопустимый тип записи")


def _v_resolvectl(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        long=frozenset({"--no-pager", "--value", "--help"}),
        long_value=frozenset({"--json"}),
    )
    parsed = _parse_flags(args, spec)
    verbs = {
        "status",
        "query",
        "statistics",
        "dns",
        "domain",
        "default-route",
        "llmnr",
        "mdns",
        "dnssec",
    }
    if not parsed.positionals or parsed.positionals[0] not in verbs:
        raise ValidationError("resolvectl: разрешены только status/query/statistics/dns")
    if len(parsed.positionals) > 3:
        raise ValidationError("resolvectl: слишком много аргументов")
    if parsed.positionals[0] == "query" and (
        len(parsed.positionals) < 2 or not _HOST_RE.match(parsed.positionals[1])
    ):
        raise ValidationError("resolvectl: укажите корректное имя для query")
    for token in parsed.positionals[2:]:
        if not (_IFACE_RE.match(token) or _DOMAIN_RE.match(token)):
            raise ValidationError("resolvectl: недопустимый аргумент")


def _v_nmcli(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("tph"),
        long=frozenset({"--terse", "--pretty", "--help", "--version"}),
    )
    parsed = _parse_flags(args, spec)
    pos = parsed.positionals
    if not pos:
        raise ValidationError("nmcli: укажите объект (general, device, connection, radio)")
    allowed_sequences = {
        ("general",),
        ("general", "status"),
        ("general", "permissions"),
        ("device",),
        ("device", "status"),
        ("device", "show"),
        ("connection",),
        ("connection", "show"),
        ("networking", "connectivity"),
        ("radio",),
        ("radio", "wifi"),
    }
    head = tuple(pos[:2])
    if head not in allowed_sequences:
        raise ValidationError("nmcli: разрешены только команды просмотра состояния")
    for token in pos[2:]:
        if not (_IFACE_RE.match(token) or re.match(r"^[A-Za-z0-9 ._:@-]{1,120}$", token)):
            raise ValidationError("nmcli: недопустимое имя интерфейса или соединения")
    if len(pos) > 3:
        raise ValidationError("nmcli: слишком много аргументов")


def _v_journalctl(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("bkrqx"),
        short_value=frozenset("nupog"),
        long=frozenset(
            {
                "--no-pager",
                "--no-hostname",
                "--utc",
                "--reverse",
                "--quiet",
                "--merge",
                "--user",
                "--system",
                "--boot",
                "--dmesg",
                "--help",
                "--version",
            }
        ),
        long_value=frozenset(
            {
                "--lines",
                "--unit",
                "--priority",
                "--since",
                "--until",
                "--grep",
                "--output",
            }
        ),
    )
    parsed = _parse_flags(args, spec)
    if parsed.positionals:
        raise ValidationError("journalctl: позиционные фильтры запрещены, используйте -u/-g")
    lines = parsed.value("-n", "--lines")
    if lines is not None:
        _int_value(lines, name="-n", minimum=1, maximum=200)
    units = parsed.values("-u", "--unit")
    if len(units) > 2:
        raise ValidationError("journalctl: не более двух unit")
    _check_units(units, limit=2)
    priority = parsed.value("-p", "--priority")
    if priority is not None:
        allowed_priority = {
            "0", "1", "2", "3", "4", "5", "6", "7",
            "emerg", "alert", "crit", "err", "warning", "notice", "info", "debug",
        }
        if priority not in allowed_priority:
            raise ValidationError("journalctl: недопустимый приоритет")
    for name in ("--since", "--until"):
        value = parsed.value(name)
        if value is not None and not _TIME_RE.match(value):
            raise ValidationError(f"journalctl: недопустимое время для {name}")
    grep = parsed.value("-g", "--grep")
    if grep is not None and (len(grep) > 80 or not _SAFE_TOKEN_RE.match(grep)):
        raise ValidationError("journalctl: недопустимый шаблон поиска")
    output = parsed.value("-o", "--output")
    if output is not None and output not in {"short", "cat", "json", "json-pretty", "verbose"}:
        raise ValidationError("journalctl: недопустимый формат вывода")


def _v_systemctl(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("aqlr"),
        short_value=frozenset("n"),
        long=frozenset(
            {
                "--no-pager",
                "--no-legend",
                "--plain",
                "--all",
                "--failed",
                "--full",
                "--recursive",
                "--value",
                "--user",
                "--system",
                "--quiet",
                "--version",
                "--help",
            }
        ),
        long_value=frozenset({"--state", "--type", "--lines"}),
    )
    parsed = _parse_flags(args, spec)
    if not parsed.positionals:
        if parsed.has("--failed", "--all", "--version", "--help"):
            return
        raise ValidationError("systemctl: укажите команду (status, list-units, is-active)")
    verbs = {
        "status",
        "list-units",
        "list-unit-files",
        "list-sockets",
        "list-timers",
        "list-jobs",
        "is-active",
        "is-enabled",
        "is-failed",
        "is-system-running",
        "get-default",
        "help",
    }
    if parsed.positionals[0] not in verbs:
        raise ValidationError("systemctl: изменяющие команды запрещены (start/stop/restart/enable)")
    _check_units(parsed.positionals[1:], limit=3)
    lines = parsed.value("-n", "--lines")
    if lines is not None:
        _int_value(lines, name="-n", minimum=1, maximum=200)
    state = parsed.value("--state")
    if state is not None and not re.match(r"^[A-Za-z0-9_,.-]{1,80}$", state):
        raise ValidationError("systemctl: недопустимое состояние")
    unit_type = parsed.value("--type")
    if unit_type is not None and not re.match(r"^[A-Za-z0-9_,.-]{1,80}$", unit_type):
        raise ValidationError("systemctl: недопустимый тип unit")


# --- Процессы, память, диски ----------------------------------------------


def _v_ps(args: list[str], ctx: _Context) -> None:
    if args == ["aux"]:
        return
    spec = FlagSpec(
        short=frozenset("efaxw"),
        short_value=frozenset("uopC"),
        long=frozenset({"--no-headers"}),
        long_value=frozenset({"--sort"}),
    )
    parsed = _parse_flags(args, spec)
    if parsed.positionals:
        raise ValidationError("ps: позиционные аргументы запрещены (кроме ps aux)")
    fields = {
        "pid", "ppid", "user", "uid", "gid", "pcpu", "pmem", "stat", "state",
        "start", "time", "etime", "comm", "args", "cmd", "rss", "vsz", "tty",
        "nice", "pri", "ni", "thcount", "%cpu", "%mem",
    }
    for value in parsed.values("-o"):
        for part in value.split(","):
            if part not in fields:
                raise ValidationError(f"ps: недопустимое поле формата «{part}»")
    sort = parsed.value("--sort")
    if sort is not None:
        cleaned = sort.lstrip("+-")
        if cleaned not in fields:
            raise ValidationError("ps: недопустимое поле сортировки")
    for value in parsed.values("-p"):
        if not _NUMBER_RE.match(value):
            raise ValidationError("ps: PID должен быть числом")
    for value in parsed.values("-u"):
        if not re.match(r"^[A-Za-z0-9_.-]{1,64}$", value):
            raise ValidationError("ps: недопустимое имя пользователя")


def _v_pgrep(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("aflnocx"),
        short_value=frozenset("u"),
        long=frozenset(
            {"--full", "--list-name", "--newest", "--oldest", "--count", "--exact"}
        ),
        long_value=frozenset({"--user"}),
    )
    parsed = _parse_flags(args, spec)
    _require_between(parsed.positionals, 1, 1, "pgrep")
    pattern = parsed.positionals[0]
    if not re.match(r"^[A-Za-z0-9_.:@ /+-]{1,100}$", pattern):
        raise ValidationError("pgrep: недопустимый шаблон")
    user = parsed.value("-u", "--user")
    if user is not None and not re.match(r"^[A-Za-z0-9_.-]{1,64}$", user):
        raise ValidationError("pgrep: недопустимое имя пользователя")


def _v_free(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("hmgbkwt"),
        long=frozenset(
            {
                "--human",
                "--mebi",
                "--gibi",
                "--kibi",
                "--wide",
                "--total",
                "--help",
                "--version",
            }
        ),
    )
    parsed = _parse_flags(args, spec)
    if parsed.positionals:
        raise ValidationError("free: аргументы не разрешены")


def _v_df(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("hHTiaPkl"),
        short_value=frozenset("tx"),
        long=frozenset(
            {
                "--human-readable",
                "--si",
                "--print-type",
                "--inodes",
                "--all",
                "--portability",
                "--local",
                "--total",
                "--help",
                "--version",
            }
        ),
        long_value=frozenset({"--type", "--exclude-type"}),
    )
    parsed = _parse_flags(args, spec)
    if len(parsed.positionals) > 3:
        raise ValidationError("df: слишком много путей")
    for path in parsed.positionals:
        _safe_any_path(path, ctx.cwd)


def _v_du(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("hsxcbmkg"),
        short_value=frozenset("d"),
        long=frozenset(
            {"--human-readable", "--summarize", "--one-file-system", "--total",
             "--apparent-size", "--bytes", "--help", "--version"}
        ),
        long_value=frozenset({"--max-depth"}),
    )
    parsed = _parse_flags(args, spec)
    _require_between(parsed.positionals, 1, 3, "du")
    depth = parsed.value("-d", "--max-depth")
    if depth is not None:
        _int_value(depth, name="--max-depth", minimum=1, maximum=3)
    if not parsed.has("-s", "--summarize") and depth is None:
        raise ValidationError("du: укажите -s или --max-depth не больше 3")
    roots = (*SAFE_PATH_ROOTS, str(Path.home()))
    for path in parsed.positionals:
        _path_in_roots(path, ctx.cwd, roots)


def _v_lsblk(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("fnJptdbl"),
        short_value=frozenset("o"),
        long=frozenset(
            {"--fs", "--json", "--paths", "--nodeps", "--topology", "--perms",
             "--bytes", "--list", "--help", "--version"}
        ),
        long_value=frozenset({"--output"}),
    )
    parsed = _parse_flags(args, spec)
    columns = {
        "NAME", "KNAME", "PATH", "MAJ:MIN", "FSTYPE", "FSVER", "LABEL", "UUID",
        "MOUNTPOINTS", "MOUNTPOINT", "TYPE", "SIZE", "RO", "RM", "MODEL", "SERIAL",
        "TRAN", "VENDOR", "REV", "STATE", "GROUP", "MODE", "OWNER", "ALIGNMENT",
        "MIN-IO", "OPT-IO", "PHY-SEC", "LOG-SEC", "ROTA", "SCHED", "RQ-SIZE",
        "DISC-ALN", "DISC-GRAN", "DISC-MAX", "DISC-ZERO", "WSAME", "WWN", "RAND",
        "PKNAME", "HCTL", "SUBSYSTEMS", "ZONED", "DAX",
    }
    for value in [*parsed.values("-o"), *parsed.values("--output")]:
        for part in value.split(","):
            if part.upper() not in columns:
                raise ValidationError(f"lsblk: недопустимая колонка «{part}»")
    for path in parsed.positionals:
        if not re.match(r"^/dev/[A-Za-z0-9/_-]{1,120}$", path):
            raise ValidationError("lsblk: недопустимое устройство")


def _v_findmnt(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("JnlDh"),
        short_value=frozenset("toT"),
        long=frozenset(
            {"--json", "--noheadings", "--list", "--df", "--verify", "--real",
             "--pseudo", "--kernel", "--help", "--version"}
        ),
        long_value=frozenset({"--type", "--output", "--target"}),
    )
    parsed = _parse_flags(args, spec)
    if len(parsed.positionals) > 2:
        raise ValidationError("findmnt: слишком много аргументов")
    for path in parsed.positionals:
        _safe_any_path(path, ctx.cwd)
    target = parsed.value("-T", "--target")
    if target is not None:
        _safe_any_path(target, ctx.cwd)


def _v_btrfs(args: list[str], ctx: _Context) -> None:
    if not args:
        raise ValidationError("btrfs: укажите подкоманду (filesystem df, subvolume list)")
    object_name = args[0]
    mapping = {
        "filesystem": "filesystem",
        "fi": "filesystem",
        "subvolume": "subvolume",
        "sub": "subvolume",
        "device": "device",
        "dev": "device",
        "scrub": "scrub",
    }
    if object_name not in mapping:
        raise ValidationError("btrfs: разрешены только filesystem/subvolume/device/scrub")
    family = mapping[object_name]
    if len(args) < 2:
        raise ValidationError("btrfs: укажите действие")
    action = args[1]
    allowed_actions = {
        "filesystem": {"df", "usage", "show"},
        "subvolume": {"list", "show"},
        "device": {"stats"},
        "scrub": {"status"},
    }
    if action not in allowed_actions[family]:
        raise ValidationError(f"btrfs: действие {action} запрещено")
    rest = args[2:]
    spec = FlagSpec(
        short=frozenset("hbkmgTRoaplstursq"),
        long=frozenset({"--raw", "--human-readable", "--help", "--version"}),
    )
    parsed = _parse_flags(rest, spec)
    if len(parsed.positionals) > 2:
        raise ValidationError("btrfs: слишком много путей")
    if family == "filesystem" and action == "show" and not parsed.positionals:
        return
    if not parsed.positionals:
        raise ValidationError("btrfs: укажите путь или устройство")
    for path in parsed.positionals:
        roots = (*SAFE_PATH_ROOTS, "/", str(Path.home()))
        if action == "df" or family == "subvolume":
            _path_in_roots(path, ctx.cwd, roots)
        else:
            _safe_any_path(path, ctx.cwd)


def _v_sensors(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("Aujflh"),
        long=frozenset(
            {"--no-adapter", "--fahrenheit", "--json", "--help", "--version"}
        ),
    )
    parsed = _parse_flags(args, spec)
    if parsed.positionals:
        raise ValidationError("sensors: аргументы не разрешены")


def _v_nvidia_smi(args: list[str], ctx: _Context) -> None:
    fields = {
        "name", "memory.total", "memory.used", "memory.free", "utilization.gpu",
        "utilization.memory", "temperature.gpu", "driver_version", "power.draw",
        "fan.speed", "clocks.current.graphics", "clocks.current.memory", "pstate",
    }
    if not args:
        return
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in {"-q", "--query", "-L", "--list-gpus", "-h", "--help"}:
            index += 1
            continue
        if arg.startswith("--query-gpu="):
            query = arg.split("=", 1)[1]
            for part in query.split(","):
                if part not in fields:
                    raise ValidationError(f"nvidia-smi: недопустимое поле «{part}»")
            index += 1
            continue
        if arg.startswith("--format="):
            fmt = arg.split("=", 1)[1]
            if fmt not in {"csv", "csv,noheader", "csv,noheader,nounits"}:
                raise ValidationError("nvidia-smi: недопустимый формат")
            index += 1
            continue
        if arg in {"-i", "--id"}:
            index += 1
            if index >= len(args) or not re.match(r"^[0-9A-Za-z, -]{1,40}$", args[index]):
                raise ValidationError("nvidia-smi: недопустимый GPU id")
            index += 1
            continue
        raise ValidationError(f"nvidia-smi: аргумент «{arg}» запрещён")


def _v_docker(args: list[str], ctx: _Context) -> str:
    if not args:
        raise ValidationError("docker: укажите подкоманду (ps, images, logs)")
    if any(arg.startswith("-") for arg in args[:1]):
        raise ValidationError("docker: глобальные флаги запрещены")
    subcommand = args[0]
    rest = args[1:]
    aliases = {"container": "container", "image": "image", "volume": "volume",
               "network": "network", "system": "system"}
    if subcommand in aliases:
        if not rest:
            raise ValidationError("docker: укажите действие после объекта")
        subcommand = f"{subcommand} {rest[0]}"
        rest = rest[1:]

    note = ""
    if subcommand in {"ps", "container ps", "container ls"}:
        spec = FlagSpec(
            short=frozenset("aq"),
            short_value=frozenset("n"),
            long=frozenset({"--all", "--quiet"}),
            long_value=frozenset({"--filter", "--last"}),
        )
        parsed = _parse_flags(rest, spec)
        if parsed.positionals:
            raise ValidationError("docker ps: позиционные аргументы запрещены")
        for value in parsed.values("--filter"):
            if not re.match(r"^[A-Za-z0-9_=.,:/@-]{1,120}$", value):
                raise ValidationError("docker ps: недопустимый фильтр")
        last = parsed.value("-n", "--last")
        if last is not None:
            _int_value(last, name="--last", minimum=1, maximum=50)
    elif subcommand in {"images", "image ls", "image list"}:
        spec = FlagSpec(
            short=frozenset("aq"),
            long=frozenset({"--all", "--quiet", "--digests"}),
            long_value=frozenset({"--filter"}),
        )
        parsed = _parse_flags(rest, spec)
        if parsed.positionals:
            raise ValidationError("docker images: позиционные аргументы запрещены")
    elif subcommand == "logs":
        spec = FlagSpec(
            short=frozenset("t"),
            short_value=frozenset("n"),
            long=frozenset({"--timestamps", "--details"}),
            long_value=frozenset({"--tail", "--since", "--until"}),
        )
        parsed = _parse_flags(rest, spec)
        _require_between(parsed.positionals, 1, 1, "docker logs")
        if not _CONTAINER_RE.match(parsed.positionals[0]):
            raise ValidationError("docker logs: недопустимое имя контейнера")
        tail = parsed.value("-n", "--tail")
        if tail is not None:
            _int_value(tail, name="--tail", minimum=1, maximum=200)
        for name in ("--since", "--until"):
            value = parsed.value(name)
            if value is not None and not _TIME_RE.match(value):
                raise ValidationError(f"docker logs: недопустимое время {name}")
        note = "Логи контейнера могут содержать секреты; они будут замаскированы."
    elif subcommand == "stats":
        spec = FlagSpec(
            short=frozenset("a"),
            long=frozenset({"--all", "--no-stream"}),
        )
        parsed = _parse_flags(rest, spec)
        if len(parsed.positionals) > 10:
            raise ValidationError("docker stats: слишком много контейнеров")
        for name in parsed.positionals:
            if not _CONTAINER_RE.match(name):
                raise ValidationError("docker stats: недопустимое имя контейнера")
        if not parsed.has("--no-stream"):
            args.append("--no-stream")
            note = "Добавлен --no-stream, чтобы команда не зависла."
    elif subcommand in {"version", "info"}:
        if rest:
            raise ValidationError(f"docker {subcommand}: аргументы запрещены")
    elif subcommand == "system df":
        if rest:
            raise ValidationError("docker system df: аргументы запрещены")
    elif subcommand in {"volume ls", "network ls"}:
        spec = FlagSpec(
            short=frozenset("q"),
            long=frozenset({"--quiet"}),
            long_value=frozenset({"--filter"}),
        )
        parsed = _parse_flags(rest, spec)
        if parsed.positionals:
            raise ValidationError(f"docker {subcommand}: позиционные аргументы запрещены")
        for value in parsed.values("--filter"):
            if not re.match(r"^[A-Za-z0-9_=.,:/@-]{1,120}$", value):
                raise ValidationError(f"docker {subcommand}: недопустимый фильтр")
    else:
        raise ValidationError(
            "docker: разрешены только ps, images, logs, stats, version, info, system df"
        )
    return note


# --- Пакеты и оборудование -------------------------------------------------


def _v_pacman(args: list[str], ctx: _Context) -> None:
    if not args:
        raise ValidationError("pacman: укажите запрос, например -Qi")
    first = args[0]
    if first.startswith("--query"):
        suffix = first[len("--query") :]
        allowed_long = {
            "", "info", "list", "owns", "search", "explicit", "foreign", "native",
            "unrequired", "deps", "check", "quiet", "upgrades",
        }
        if suffix.lstrip("-") not in allowed_long:
            raise ValidationError("pacman: разрешены только запросы --query")
    elif first.startswith("-Q"):
        flags = first[2:]
        if any(char not in "Qietlsonmudqckp" for char in flags):
            raise ValidationError("pacman: разрешены только запросы -Q")
    else:
        raise ValidationError("pacman: установка и удаление пакетов запрещены")
    for arg in args[1:]:
        if arg.startswith("-"):
            raise ValidationError("pacman: дополнительные флаги запрещены")
        if arg.startswith("/"):
            _safe_any_path(arg, ctx.cwd)
        elif not _PACKAGE_RE.match(arg):
            raise ValidationError(f"pacman: недопустимое имя пакета «{arg}»")
    if len(args) > 6:
        raise ValidationError("pacman: слишком много аргументов")


def _v_lspci(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("kntvmD"),
        short_value=frozenset("sd"),
        long=frozenset(
            {"--kernel", "--numeric", "--tree", "--verbose", "--machine", "--domain"}
        ),
    )
    parsed = _parse_flags(args, spec)
    if parsed.positionals:
        raise ValidationError("lspci: позиционные аргументы запрещены")


def _v_lsusb(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("tvV"),
        short_value=frozenset("ds"),
        long=frozenset({"--tree", "--verbose"}),
    )
    parsed = _parse_flags(args, spec)
    if parsed.positionals:
        raise ValidationError("lsusb: позиционные аргументы запрещены")


def _v_uname(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("armsnovpi"),
        long=frozenset(
            {"--all", "--kernel-release", "--machine", "--kernel-name",
             "--nodename", "--operating-system", "--kernel-version", "--processor",
             "--hardware-platform"}
        ),
    )
    parsed = _parse_flags(args, spec)
    if parsed.positionals:
        raise ValidationError("uname: аргументы не разрешены")


def _v_uptime(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("pshV"),
        long=frozenset({"--pretty", "--since", "--help", "--version"}),
    )
    parsed = _parse_flags(args, spec)
    if parsed.positionals:
        raise ValidationError("uptime: аргументы не разрешены")


def _v_whoami(args: list[str], ctx: _Context) -> None:
    if args:
        raise ValidationError("whoami: аргументы не разрешены")


def _v_id(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("ugGnrz"),
        long=frozenset({"--user", "--group", "--groups", "--name", "--real"}),
    )
    parsed = _parse_flags(args, spec)
    if parsed.positionals:
        raise ValidationError("id: имена пользователей не разрешены")


def _v_hostnamectl(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("h"),
        long=frozenset(
            {"--no-pager", "--static", "--transient", "--pretty", "--help", "--version"}
        ),
    )
    parsed = _parse_flags(args, spec)
    if parsed.positionals and parsed.positionals != ["status"]:
        raise ValidationError("hostnamectl: разрешён только status")


def _v_timedatectl(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("h"),
        long=frozenset({"--no-pager", "--value", "--all", "--help", "--version"}),
    )
    parsed = _parse_flags(args, spec)
    allowed = {"status", "show", "timesync-status", "list-timezones"}
    if parsed.positionals and parsed.positionals[0] not in allowed:
        raise ValidationError("timedatectl: изменяющие команды запрещены")
    if len(parsed.positionals) > 1:
        raise ValidationError("timedatectl: слишком много аргументов")


def _v_getent(args: list[str], ctx: _Context) -> None:
    if args and args[0].startswith("-"):
        raise ValidationError("getent: флаги запрещены")
    _require_between(args, 2, 2, "getent")
    database, key = args
    if database not in {"hosts", "ahosts", "services", "protocols", "networks"}:
        raise ValidationError("getent: разрешены hosts/services/protocols/networks")
    if not (_HOST_RE.match(key) or _SAFE_TOKEN_RE.match(key)):
        raise ValidationError("getent: недопустимый ключ")


# --- Файлы -----------------------------------------------------------------


def _v_stat(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("Lft"),
        long=frozenset({"--dereference", "--file-system", "--terse"}),
    )
    parsed = _parse_flags(args, spec)
    _require_between(parsed.positionals, 1, 3, "stat")
    for path in parsed.positionals:
        _safe_any_path(path, ctx.cwd)


def _v_file(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("biLsk"),
        long=frozenset({"--brief", "--mime", "--dereference", "--special-files",
                        "--keep-going"}),
    )
    parsed = _parse_flags(args, spec)
    _require_between(parsed.positionals, 1, 3, "file")
    for path in parsed.positionals:
        _safe_any_path(path, ctx.cwd)


def _v_cat(args: list[str], ctx: _Context) -> None:
    if args and args[0].startswith("-"):
        raise ValidationError("cat: флаги запрещены")
    _require_between(args, 1, 3, "cat")
    for path in args:
        _safe_read_file(path, ctx.cwd)


def _v_head_tail(args: list[str], ctx: _Context, *, name: str) -> None:
    spec = FlagSpec(
        short=frozenset("qv"),
        short_value=frozenset("nc"),
        long=frozenset({"--quiet", "--verbose"}),
        long_value=frozenset({"--lines", "--bytes"}),
    )
    parsed = _parse_flags(args, spec)
    _require_between(parsed.positionals, 1, 3, name)
    lines = parsed.value("-n", "--lines")
    if lines is not None:
        _int_value(lines, name="-n", minimum=1, maximum=200)
    count = parsed.value("-c", "--bytes")
    if count is not None:
        _int_value(count, name="-c", minimum=1, maximum=20000)
    for path in parsed.positionals:
        _safe_read_file(path, ctx.cwd)


def _v_ls(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("lahtSr1dFinogp"),
        long=frozenset(
            {"--long", "--all", "--human-readable", "--directory", "--classify",
             "--inode", "--numeric-uid-gid", "--help", "--version"}
        ),
    )
    parsed = _parse_flags(args, spec)
    if len(parsed.positionals) > 3:
        raise ValidationError("ls: слишком много путей")
    for path in parsed.positionals:
        _path_in_roots(path, ctx.cwd, SAFE_PATH_ROOTS)


def _v_sysctl(args: list[str], ctx: _Context) -> None:
    if args and args[0] in {"-w", "--write", "-a", "--all"}:
        raise ValidationError("sysctl: запись и полный дамп запрещены")
    spec = FlagSpec(
        short=frozenset("n"),
        long=frozenset({"--values", "--help", "--version"}),
    )
    parsed = _parse_flags(args, spec)
    _require_between(parsed.positionals, 1, 5, "sysctl")
    for key in parsed.positionals:
        if "=" in key or not re.match(r"^[a-z0-9_.-]{1,120}$", key):
            raise ValidationError("sysctl: недопустимое имя параметра")


def _v_lsmod(args: list[str], ctx: _Context) -> None:
    if args:
        raise ValidationError("lsmod: аргументы не разрешены")


def _v_modinfo(args: list[str], ctx: _Context) -> None:
    _require_between(args, 1, 3, "modinfo")
    for name in args:
        if not re.match(r"^[A-Za-z0-9_.-]{1,120}$", name):
            raise ValidationError("modinfo: недопустимое имя модуля")


def _v_ethtool(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("ikSglacT"),
        long=frozenset(
            {"--driver", "--show-features", "--statistics", "--show-ring",
             "--show-channels", "--show-pause", "--show-coalesce",
             "--show-time-stamping"}
        ),
    )
    parsed = _parse_flags(args, spec)
    _require_between(parsed.positionals, 1, 1, "ethtool")
    if not _IFACE_RE.match(parsed.positionals[0]):
        raise ValidationError("ethtool: недопустимое имя интерфейса")


def _v_iw(args: list[str], ctx: _Context) -> None:
    if not args:
        return
    if args[0] == "list":
        if len(args) != 1:
            raise ValidationError("iw list: аргументы запрещены")
        return
    if args[0] == "phy":
        if len(args) == 1:
            return
        if len(args) == 3 and args[1] in {"phy0", "phy1", "phy2", "phy3"} and args[2] == "info":
            return
        raise ValidationError("iw: недопустимая команда phy")
    if args[0] != "dev":
        raise ValidationError("iw: разрешены только dev/phy/list")
    if len(args) == 1:
        return
    if not _IFACE_RE.match(args[1]):
        raise ValidationError("iw: недопустимое имя интерфейса")
    if len(args) == 3 and args[2] in {"info", "link"}:
        return
    if args[2:] == ["station", "dump"]:
        return
    raise ValidationError("iw: разрешены только info/link/station dump")


def _v_iwconfig(args: list[str], ctx: _Context) -> None:
    if len(args) > 1:
        raise ValidationError("iwconfig: допускается один интерфейс")
    if args and not _IFACE_RE.match(args[0]):
        raise ValidationError("iwconfig: недопустимое имя интерфейса")


def _v_rfkill(args: list[str], ctx: _Context) -> None:
    if not args or args[0] != "list":
        raise ValidationError("rfkill: разрешена только команда list")
    if len(args) == 2 and args[1] not in {
        "all", "wifi", "wlan", "bluetooth", "uwb", "gps", "fm", "nfc", "wwan"
    }:
        raise ValidationError("rfkill: недопустимый тип")
    if len(args) > 2:
        raise ValidationError("rfkill: слишком много аргументов")


def _v_lsof(args: list[str], ctx: _Context) -> None:
    spec = FlagSpec(
        short=frozenset("inPatU"),
        short_value=frozenset("upci"),
        long=frozenset({"--inet", "--numeric", "--no-port", "--all", "--terse", "--unix"}),
        long_value=frozenset({"--user", "--pid", "--command"}),
    )
    parsed = _parse_flags(args, spec, )
    if parsed.positionals:
        raise ValidationError("lsof: позиционные аргументы запрещены")
    for value in parsed.values("-i", "--inet"):
        if value and not re.match(r"^[A-Za-z0-9:.*-]{1,120}$", value):
            raise ValidationError("lsof: недопустимый сетевой фильтр")
    for value in parsed.values("-p", "--pid"):
        if not _NUMBER_RE.match(value):
            raise ValidationError("lsof: PID должен быть числом")


def _v_systemd_analyze(args: list[str], ctx: _Context) -> None:
    if not args:
        raise ValidationError("systemd-analyze: укажите blame/critical-chain/time/verify")
    verbs = {"blame", "critical-chain", "time", "verify"}
    if args[0] not in verbs:
        raise ValidationError("systemd-analyze: разрешены blame/critical-chain/time/verify")
    rest = args[1:]
    if len(rest) > 1:
        raise ValidationError("systemd-analyze: слишком много аргументов")
    for unit in rest:
        if not _UNIT_RE.match(unit):
            raise ValidationError("systemd-analyze: недопустимое имя unit")


# --- Регистрация -----------------------------------------------------------


_VALIDATORS: dict[str, Callable[[list[str], _Context], str | None]] = {
    "ss": _v_ss,
    "ip": _v_ip,
    "ping": _v_ping,
    "dig": _v_dig,
    "resolvectl": _v_resolvectl,
    "nmcli": _v_nmcli,
    "journalctl": _v_journalctl,
    "systemctl": _v_systemctl,
    "ps": _v_ps,
    "pgrep": _v_pgrep,
    "free": _v_free,
    "df": _v_df,
    "du": _v_du,
    "lsblk": _v_lsblk,
    "findmnt": _v_findmnt,
    "btrfs": _v_btrfs,
    "sensors": _v_sensors,
    "nvidia-smi": _v_nvidia_smi,
    "docker": _v_docker,
    "pacman": _v_pacman,
    "lspci": _v_lspci,
    "lsusb": _v_lsusb,
    "uname": _v_uname,
    "uptime": _v_uptime,
    "whoami": _v_whoami,
    "id": _v_id,
    "hostnamectl": _v_hostnamectl,
    "timedatectl": _v_timedatectl,
    "getent": _v_getent,
    "stat": _v_stat,
    "file": _v_file,
    "cat": _v_cat,
    "head": lambda args, ctx: _v_head_tail(args, ctx, name="head"),
    "tail": lambda args, ctx: _v_head_tail(args, ctx, name="tail"),
    "ls": _v_ls,
    "sysctl": _v_sysctl,
    "lsmod": _v_lsmod,
    "modinfo": _v_modinfo,
    "ethtool": _v_ethtool,
    "iw": _v_iw,
    "iwconfig": _v_iwconfig,
    "rfkill": _v_rfkill,
    "lsof": _v_lsof,
    "systemd-analyze": _v_systemd_analyze,
}

assert set(_VALIDATORS) <= ALLOWED_BINARIES, "Валидатор отсутствует для разрешённой команды"
for _name in ALLOWED_BINARIES:
    assert _name in _VALIDATORS, f"Нет валидатора для {_name}"


@dataclass(frozen=True)
class _Context:
    cwd: Path


def _safe_which(name: str) -> str | None:
    if "/" in name or "\\" in name:
        return None
    for directory in SAFE_EXEC_DIRS:
        candidate = Path(directory) / name
        try:
            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                continue
            stat = candidate.stat()
        except OSError:
            continue
        if stat.st_uid != 0:
            continue
        if stat.st_mode & 0o022:
            continue
        try:
            return str(candidate.resolve())
        except OSError:
            continue
    return None


def _sanitize_argv(argv: list[str]) -> None:
    if not argv:
        raise ValidationError("Пустая команда")
    if len(argv) > _MAX_ARGS:
        raise ValidationError("Слишком много аргументов")
    total = 0
    for arg in argv:
        if not isinstance(arg, str):
            raise ValidationError("Аргументы команды должны быть строками")
        if not arg:
            raise ValidationError("Пустой аргумент команды запрещён")
        if len(arg) > _MAX_ARG_LEN:
            raise ValidationError("Слишком длинный аргумент команды")
        total += len(arg)
        for char in arg:
            if char in SHELL_METACHARACTERS or char in GLOB_CHARACTERS:
                raise ValidationError(
                    "Обнаружен shell-метасимвол или глоб. Цепочки и подстановки запрещены."
                )
            if ord(char) < 32 or ord(char) == 127:
                raise ValidationError("Управляющие символы в аргументах запрещены")
    if total > _MAX_TOTAL_LEN:
        raise ValidationError("Слишком длинная команда")


_RISK_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def _max_risk(*values: str) -> RiskLevel:
    best = max(values, key=lambda item: _RISK_ORDER.get(item, 0))
    return best if best in _RISK_ORDER else "low"  # type: ignore[return-value]


class CommandValidator:
    def __init__(
        self,
        *,
        allow_sudo: bool = True,
        cwd: Path | None = None,
        resolver: Callable[[str], str | None] | None = None,
    ):
        self.allow_sudo = allow_sudo
        self.cwd = (cwd or Path.cwd()).resolve()
        self.resolver = resolver or _safe_which

    def validate(self, raw: ProposedCommand) -> ValidatedCommand:
        argv = [arg for arg in raw.argv]
        _sanitize_argv(argv)

        binary = argv[0]
        if not _BINARY_RE.match(binary):
            raise ValidationError(
                "Имя программы должно быть простым именем без пути и пробелов"
            )
        if binary in DENIED_BINARIES:
            raise ValidationError(f"Команда «{binary}» запрещена политикой безопасности")
        if binary not in ALLOWED_BINARIES:
            raise ValidationError(
                f"Команда «{binary}» не входит в белый список Ryoku AI Terminal"
            )

        validator = _VALIDATORS.get(binary)
        if validator is None:
            raise ValidationError(f"Для «{binary}» нет безопасного валидатора")

        args = argv[1:]
        extra_note = validator(args, _Context(cwd=self.cwd)) or ""
        argv = [binary, *args]

        executable = self.resolver(binary)
        if not executable:
            raise ValidationError(f"Утилита «{binary}» не найдена в системных каталогах")
        if "/" not in executable:
            raise ValidationError("Не удалось получить абсолютный путь к утилите")

        requires_sudo = bool(raw.requires_sudo)
        if requires_sudo:
            if not self.allow_sudo:
                raise ValidationError("Повышение прав отключено в конфигурации")
            if binary not in SUDO_ALLOWED:
                requires_sudo = False
                extra_note = (
                    extra_note + " " if extra_note else ""
                ) + "sudo для этой утилиты не требуется и не будет использован."

        risk = _max_risk(
            raw.risk,
            "medium" if binary in {"journalctl", "docker", "btrfs"} else "low",
            "medium" if requires_sudo else "low",
        )
        description = raw.description.strip()[:500] or policy_note(binary)
        note = policy_note(binary)
        if extra_note:
            note = f"{note} {extra_note}".strip()

        digest = compute_digest(
            argv=tuple(argv),
            executable=executable,
            requires_sudo=requires_sudo,
        )
        return ValidatedCommand(
            argv=tuple(argv),
            executable=executable,
            description=description,
            risk=risk,
            requires_sudo=requires_sudo,
            policy_note=note,
            digest=digest,
        )

    def revalidate(self, command: ValidatedCommand) -> ValidatedCommand:
        raw = ProposedCommand(
            argv=list(command.argv),
            description=command.description,
            risk=command.risk,
            requires_sudo=command.requires_sudo,
        )
        fresh = self.validate(raw)
        if fresh.digest != command.digest or fresh.executable != command.executable:
            raise ValidationError(
                "Команда изменилась после подтверждения — выполнение отменено",
                category="integrity",
            )
        return fresh

    def validate_line(self, line: str, *, description: str = "") -> ValidatedCommand:
        try:
            parts = shlex.split(line, posix=True)
        except ValueError as exc:
            raise ValidationError(f"Не удалось разобрать команду: {exc}") from exc
        if not parts:
            raise ValidationError("Пустая команда")
        raw = ProposedCommand(argv=parts, description=description)
        return self.validate(raw)
