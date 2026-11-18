"""CLI Ryoku AI Terminal на Typer + Rich."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import typer
from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import __version__
from .ai.base import (
    AIError,
    AIMessage,
    AllProvidersFailed,
    ChatRequest,
    ChatResponse,
    NoProvidersConfigured,
)
from .ai.router import PROVIDER_CLASSES, Router
from .config import (
    PROVIDER_NAMES,
    Config,
    ConfigError,
    Credentials,
    CredentialsError,
    config_path,
    credentials_path,
    credentials_permission_issue,
    history_db_path,
    load_config,
    load_credentials,
    save_config,
    save_credentials,
    state_dir,
    validate_secret_format,
)
from .context import collect_context
from .conversation import ConversationSession
from .execution.approval import ApprovalEngine
from .execution.cleanup import (
    CleanupActionValidator,
    CleanupExecutor,
    CleanupResult,
    ValidatedCleanupAction,
)
from .execution.executor import ExecutionResult, Executor
from .execution.models import (
    AIAction,
    ProposedCommand,
    ValidatedCommand,
    ValidationError,
)
from .execution.planner import (
    build_analysis_prompt,
    build_explain_prompt,
    build_system_prompt,
    parse_action,
)
from .execution.validator import CommandValidator
from .redaction import contains_redaction, redact_text
from .storage.database import HistoryError
from .storage.history import HistoryService
from .system._run import format_bytes
from .system._run import which as find_tool
from .system.inspector import SystemReport, collect_system
from .system.network import NetworkReport, collect_network
from .system.processes import ProcessesReport, collect_processes
from .system.services import DockerInfo, ServiceReport, collect_services
from .system.storage import StorageReport, collect_storage
from .system.temp_inspector import (
    DirectoryInspection,
    inspect_directory,
)

APP_HELP = """Ryoku AI Terminal — AI-ассистент для терминала CachyOS/Arch Linux.

Примеры:
  ai почему медленный интернет
  ai проверь состояние docker
  ai найди процессы которые потребляют интернет
  ai объясни эту ошибку
  ai создай проект Next.js с TypeScript
  ai освободи место на диске

Диагностика без ИИ:
  ai --network        сеть и соединения
  ai --system         CPU, RAM, диски, температуры
  ai --processes      процессы
  ai --docker         Docker и Steam
  ai --storage        диски, Btrfs, крупные каталоги
  ai --history        история
  ai --models         доступные модели
  ai --config         конфигурация
  ai --chat           интерактивный чат

Подкоманды:
  ryoku-ai setup                     настроить бесплатные AI-провайдеры
  ryoku-ai models [--provider NAME]  доступные модели
  ryoku-ai history [--limit N]       история (--show ID, --clear, --yes)
  ryoku-ai config [--init]           показать или создать конфигурацию
  ryoku-ai doctor                    проверить окружение

Первый запуск: ryoku-ai setup
"""

ACCENT = "bright_cyan"
ACCENT_SOFT = "cyan"
MUTED = "dim"
DANGER = "red"
WARN = "yellow"
OK = "green"


@dataclass(slots=True)
class Runtime:
    config: Config
    credentials: Credentials
    console: Console
    history: HistoryService
    secrets: list[str]


def _make_console(config: Config | None = None) -> Console:
    no_color = bool(config and config.theme in {"mono", "accessible"})
    return Console(
        no_color=no_color,
        highlight=False,
        soft_wrap=False,
        emoji=False,
    )


def _bootstrap_console() -> Console:
    return _make_console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"Ryoku AI Terminal {__version__}")
        raise typer.Exit()


def _load_runtime(console: Console | None = None) -> Runtime:
    console = console or _bootstrap_console()
    try:
        config = load_config()
    except ConfigError as exc:
        _error_panel(console, "Ошибка конфигурации", str(exc), ["Проверьте config.toml"])
        raise typer.Exit(1) from exc
    try:
        credentials = load_credentials()
    except CredentialsError as exc:
        _error_panel(console, "Ошибка файла ключей", str(exc), [f"Файл: {credentials_path()}"])
        raise typer.Exit(1) from exc

    console = _make_console(config)
    permission_issue = credentials_permission_issue()
    if permission_issue:
        console.print(f"[{WARN}]⚠ {permission_issue}[/{WARN}]")

    secrets = [
        value
        for value in (
            credentials.groq,
            credentials.openrouter,
        )
        if value
    ]
    try:
        history = HistoryService(
            history_db_path(),
            enabled=config.history_enabled,
            secrets=secrets,
        )
    except (OSError, HistoryError) as exc:
        _error_panel(console, "История недоступна", str(exc), ["Проверьте права на ~/.local/state"])
        raise typer.Exit(1) from exc
    return Runtime(config, credentials, console, history, secrets)


def _error_panel(
    console: Console,
    title: str,
    message: str,
    hints: list[str] | None = None,
) -> None:
    body = Text()
    body.append(message)
    if hints:
        body.append("\n\n")
        for hint in hints:
            body.append(f"• {hint}\n", style=MUTED)
    console.print(Panel(body, title=Text(title, style=f"bold white on {DANGER}"), border_style=DANGER))


def _warn_panel(console: Console, title: str, message: str) -> None:
    console.print(
        Panel(
            Text(message),
            title=Text(title, style=f"bold black on {WARN}"),
            border_style=WARN,
        )
    )


def _ai_error(console: Console, exc: AIError) -> None:
    if isinstance(exc, NoProvidersConfigured):
        _error_panel(
            console,
            "AI не настроен",
            str(exc),
            [
                "Запустите: ryoku-ai setup",
                "Ключи хранятся в ~/.config/ryoku-ai/credentials.env (chmod 600)",
                "Бесплатные квоты ограничены; платная подписка не требуется.",
            ],
        )
        return
    if isinstance(exc, AllProvidersFailed):
        hints = exc.summary_lines()
        _error_panel(
            console,
            "Провайдеры недоступны",
            "Не удалось получить ответ ни от одного провайдера.",
            hints or ["Проверьте подключение к интернету и API-ключи."],
        )
        return
    _error_panel(console, "Ошибка AI", str(exc))


def _banner(console: Console) -> None:
    body = Text()
    body.append("RYOKU AI TERMINAL", style=f"bold {ACCENT}")
    body.append(f"  v{__version__}\n", style=MUTED)
    body.append("AI-ассистент для CachyOS · Fish · Hyprland", style=MUTED)
    console.print(Panel(body, border_style=ACCENT_SOFT, padding=(0, 2)))


def _provider_table(runtime: Runtime) -> Table:
    table = Table(
        title="AI-провайдеры",
        box=box.SIMPLE_HEAD,
        header_style=f"bold {ACCENT}",
        expand=False,
    )
    table.add_column("Провайдер")
    table.add_column("Ключ")
    table.add_column("Модель")
    table.add_column("Статус")
    for name in PROVIDER_NAMES:
        key = getattr(runtime.credentials, name)
        configured = "✓ настроен" if key else "— нет"
        model = runtime.config.provider_models.get(name) or runtime.config.model or "авто"
        if runtime.config.provider == name:
            status = "по умолчанию"
        elif runtime.config.provider == "auto":
            status = "резерв"
        else:
            status = "резерв"
        style = OK if key else MUTED
        table.add_row(
            Text(name, style=style),
            Text(configured, style=style),
            Text(model, style=MUTED),
            Text(status, style=MUTED),
        )
    return table


def _dashboard(runtime: Runtime) -> None:
    console = runtime.console
    _banner(console)
    console.print(_provider_table(runtime))

    stats = runtime.history.stats()
    info = Table(box=box.SIMPLE, show_header=False)
    info.add_column("", style=MUTED)
    info.add_column("")
    info.add_row("Конфигурация", Text(str(config_path())))
    info.add_row("История", Text(str(history_db_path())))
    info.add_row(
        "Записей",
        Text(
            f"{stats['conversations']} диалогов · {stats['messages']} сообщений · "
            f"{stats['commands']} команд"
        ),
    )
    console.print(info)

    if not runtime.credentials.has_any():
        _warn_panel(
            console,
            "Нужен API-ключ",
            "Ни один провайдер не настроен. Запустите: ryoku-ai setup\n"
            "Добавьте GROQ_API_KEY и/или OPENROUTER_API_KEY.",
        )
    else:
        console.print(
            f"\n[{MUTED}]Запрос:[/{MUTED}]  ai почему медленный интернет\n"
            f"[{MUTED}]Чат:[/{MUTED}]     ai --chat\n"
            f"[{MUTED}]Модели:[/{MUTED}]  ai --models\n"
            f"[{MUTED}]Диагностика:[/{MUTED}] ai --network · ai --system · ai --processes"
        )


# --- Рендеринг AI ----------------------------------------------------------


def _render_response(console: Console, message: str, response: ChatResponse) -> None:
    title = Text()
    title.append("RYOKU AI", style=f"bold {ACCENT}")
    title.append(f"  ·  {response.provider}  ·  {response.model}", style=MUTED)
    console.print(
        Panel(
            Markdown(message),
            title=title,
            border_style=ACCENT_SOFT,
            padding=(1, 2),
        )
    )
    if response.usage:
        tokens = response.usage.get("total_tokens")
        if tokens:
            console.print(f"[{MUTED}]Токенов: {tokens}[/{MUTED}]")
    for note in response.route_notes:
        console.print(f"[{WARN}]{note}[/{WARN}]")


def _render_blocked(console: Console, command: ProposedCommand, reason: str) -> None:
    body = Text()
    body.append("Команда отклонена политикой безопасности.\n\n", style=f"bold {DANGER}")
    body.append("Предложенная команда:\n", style="bold")
    body.append(" ".join(command.argv), style=MUTED)
    body.append("\n\nПричина:\n", style="bold")
    body.append(reason)
    console.print(
        Panel(
            body,
            title=Text("ЗАБЛОКИРОВАНО", style="bold white on red"),
            border_style=DANGER,
        )
    )


def _render_planned(console: Console, command: ValidatedCommand) -> None:
    body = Text()
    body.append(command.description + "\n\n")
    body.append(command.display() + "\n", style=f"bold {ACCENT}")
    body.append("\n" + command.policy_note, style=MUTED)
    console.print(
        Panel(
            body,
            title=Text("ПЛАН (выполнение отключено)", style=f"bold black on {WARN}"),
            border_style=WARN,
        )
    )


def _render_execution(
    console: Console,
    command: ValidatedCommand,
    result: ExecutionResult,
    output: str,
) -> None:
    status_text = {
        "success": "SUCCESS",
        "partial_success": "PARTIAL SUCCESS",
        "failed": "FAILED",
        "cancelled": "CANCELLED",
        "timeout": "TIMEOUT",
    }[result.status]
    if result.status == "success":
        status = Text(status_text, style=f"bold black on {OK}")
        border = OK
    elif result.status == "partial_success":
        status = Text(status_text, style="bold black on yellow")
        border = WARN
    elif result.status == "timeout":
        status = Text(status_text, style="bold black on red")
        border = DANGER
    elif result.status == "cancelled":
        status = Text(status_text, style=f"bold black on {MUTED}")
        border = MUTED
    else:
        status = Text(status_text, style="bold black on red")
        border = DANGER

    header = Text(command.display(), style=f"bold {ACCENT}", overflow="fold")
    meta = Text()
    meta.append(f"Status: {result.status}  ·  Exit code: {result.exit_code}  ·  ", style=MUTED)
    meta.append(f"{result.duration_ms} ms", style=MUTED)
    if result.truncated:
        meta.append("  ·  вывод обрезан", style=WARN)
    if contains_redaction(output):
        meta.append("  ·  секреты замаскированы", style=WARN)

    body: list[Any] = [Text("Command: ", style="bold") + header, meta]
    if result.summary:
        body.append(Text(f"Result: {result.summary}", style="bold"))
    for warning in result.warnings:
        body.append(Text(f"Warning: {warning}", style=WARN))
    if result.stdout.strip():
        display_output = result.stdout if len(result.stdout) <= 6000 else result.stdout[:6000] + "\n…"
        body.append(
            Panel(
                Syntax(display_output, "text", theme="monokai", word_wrap=True),
                border_style=MUTED,
                padding=(0, 1),
                title="stdout",
            )
        )
    else:
        body.append(Text("stdout: нет вывода.", style=MUTED))
    if result.stderr.strip():
        stderr_lines = result.stderr.splitlines()
        permission_lines = [
            line for line in stderr_lines
            if re.search(r"permission denied|operation not permitted|access denied", line, re.I)
        ]
        other_lines = [line for line in stderr_lines if line not in permission_lines]
        if permission_lines:
            body.append(Text(f"stderr: {len(permission_lines)} permission warning(s) summarized above.", style=WARN))
        if other_lines:
            display_stderr = "\n".join(other_lines[:5])
            omitted = len(other_lines) - 5
            if omitted > 0:
                display_stderr += f"\n… ещё {omitted} строк(и), полный вывод сохранён в истории"
            body.append(
                Panel(
                    Syntax(display_stderr, "text", theme="monokai", word_wrap=True),
                    border_style=WARN,
                    padding=(0, 1),
                    title="stderr",
                )
            )
    if result.error and result.error != result.summary:
        body.append(Text(result.error, style=DANGER))
    console.print(
        Panel(
            Group(*body),
            title=Text("RYOKU AI — EXECUTION RESULT · ") + status,
            border_style=border,
            padding=(1, 2),
        )
    )


def _render_cleanup_result(
    console: Console,
    action: ValidatedCleanupAction,
    result: CleanupResult,
) -> None:
    body = Text()
    body.append(f"Операция: {action.title}\n", style="bold")
    body.append(f"Команда: {result.command}\n")
    body.append(f"Статус: {result.status.replace('_', ' ').upper()}\n")
    body.append(f"Код завершения: {result.exit_code}\n")
    body.append(f"Время: {result.duration_ms} мс\n")
    if result.free_before is not None and result.free_after is not None:
        delta = result.free_after - result.free_before
        body.append(f"Свободное место: {delta:+,} байт\n".replace(",", " "))
    if result.timed_out:
        body.append("Таймаут: выполнение остановлено.\n", style=DANGER)
    if result.error:
        body.append(f"Ошибка: {result.error}\n", style=DANGER)
    try:
        structured = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        structured = None
    structured_reports = structured if isinstance(structured, list) else [structured]
    for report in structured_reports:
        if not isinstance(report, dict):
            continue
        if report.get("readable_size"):
            body.append(f"Доступный размер: {report['readable_size']}\n", style="bold")
        if report.get("inaccessible_directories"):
            body.append(
                f"Недоступные каталоги: {report['inaccessible_directories']}\n",
                style=WARN,
            )
        for warning in report.get("warnings", []):
            body.append(f"Предупреждение: {warning}\n", style=WARN)
    for label, value in (("stdout", result.stdout), ("stderr", result.stderr)):
        if value.strip():
            body.append(f"\n{label}:\n", style="bold")
            body.append(value[:6000] + ("\n…" if len(value) > 6000 else ""))
    console.print(
        Panel(
            body,
            title=Text(
                f"ACTION RESULT — {result.status.replace('_', ' ').upper()}",
                style=(
                    "bold black on green"
                    if result.status == "success"
                    else "bold black on yellow"
                    if result.status == "partial_success"
                    else "bold white on red"
                ),
            ),
            border_style=(
                OK if result.status == "success"
                else WARN if result.status == "partial_success"
                else DANGER
            ),
        )
    )


def _render_action_plan(
    console: Console,
    task: str,
    actions: list[ValidatedCleanupAction],
) -> None:
    body = Text()
    body.append(f"Задача: {task.strip() or 'План системной очистки'}\n\n", style="bold")
    body.append("Действия:\n", style="bold")
    for index, action in enumerate(actions, start=1):
        body.append(f"[{index}] {action.title}\n", style="bold bright_cyan")
        body.append(f"    {action.expected_effect}\n", style=MUTED)
        body.append(f"    Риск: {action.risk_level}\n", style=MUTED)
    console.print(
        Panel(
            body,
            title=Text("RYOKU AI — ACTION PLAN", style="bold white on dark_red"),
            border_style=ACCENT_SOFT,
            padding=(1, 2),
        )
    )


def _display_action(console: Console, action: AIAction) -> None:
    for rejected in action.rejected:
        console.print(f"[{WARN}]⚠ {rejected}[/{WARN}]")


# --- AI-запросы ------------------------------------------------------------


def _build_router(runtime: Runtime) -> Router:
    return Router(runtime.config, runtime.credentials)


def _ask_ai(
    console: Console,
    router: Router,
    request: ChatRequest,
    *,
    provider: str | None,
    model: str | None,
) -> ChatResponse | None:
    with console.status(f"[{ACCENT}]AI думает…[/{ACCENT}]", spinner="dots"):
        try:
            return asyncio.run(router.complete(request, provider=provider, model=model))
        except AIError as exc:
            _ai_error(console, exc)
            return None
        except Exception as exc:  # noqa: BLE001 - CLI не должен падать трейсбеком
            message = redact_text(
                f"{type(exc).__name__}: {exc}",
                extra_secrets=[
                    key
                    for key in (router.credentials.groq, router.credentials.openrouter)
                    if key
                ],
            )
            _error_panel(console, "Неожиданная ошибка AI", message)
            return None


def _run_commands(
    runtime: Runtime,
    session: ConversationSession,
    action: AIAction,
    *,
    allow_exec: bool = True,
) -> list[tuple[ValidatedCommand, ExecutionResult, str]]:
    console = runtime.console
    planned = action.limited(runtime.config.max_commands, maximum_actions=5)
    for rejected in planned.rejected:
        console.print(f"[{WARN}]⚠ {rejected}[/{WARN}]")
    if not planned.commands and not planned.actions:
        return []

    if planned.actions:
        action_validator = CleanupActionValidator()
        cleanup_executor = CleanupExecutor(
            action_validator,
            timeout=runtime.config.command_timeout,
        )
        action_approval = ApprovalEngine(
            console,
            CommandValidator(allow_sudo=runtime.config.allow_sudo, cwd=Path.cwd()),
            allow_sudo=runtime.config.allow_sudo,
        )
        completed_inspections: set[str] = set()
        validated_actions: list[tuple[int, ValidatedCleanupAction]] = []
        for item in planned.actions:
            try:
                validated_actions.append(
                    (len(validated_actions) + 1, action_validator.validate(item))
                )
            except ValidationError as exc:
                console.print(f"[{DANGER}]Действие отклонено: {exc.reason}[/{DANGER}]")
        task = next(
            (message.content for message in reversed(session.messages) if message.role == "user"),
            planned.message,
        )
        if validated_actions:
            _render_action_plan(
                console,
                task,
                [item for _, item in validated_actions],
            )
        if not allow_exec:
            console.print(
                Panel(
                    Text("План очистки подготовлен, выполнение отключено."),
                    title=Text("RYOKU AI — ACTION PLAN", style="bold black on yellow"),
                    border_style=WARN,
                )
            )
        else:
            for index, validated_action in validated_actions:
                outcome = action_approval.request_action_approval(
                    validated_action,
                    action_validator,
                    index=index,
                    total=len(validated_actions),
                    task=task,
                )
                if outcome.stop_plan:
                    return []
                if not outcome.approved or outcome.action is None:
                    continue
                result = cleanup_executor.execute(
                    outcome.action,
                    sudo_approved=outcome.action.requires_sudo,
                    completed_inspections=completed_inspections,
                )
                result.stdout = redact_text(result.stdout, extra_secrets=runtime.secrets)
                result.stderr = redact_text(result.stderr, extra_secrets=runtime.secrets)
                if result.error:
                    result.error = redact_text(result.error, extra_secrets=runtime.secrets)
                _render_cleanup_result(console, outcome.action, result)
                if result.ok:
                    completed_inspections.add(outcome.action.operation)
                try:
                    runtime.history.add_command(
                        session.conversation_id,
                        argv=[outcome.action.executable, *outcome.action.arguments],
                        description=outcome.action.description,
                        risk=outcome.action.risk_level,
                        requires_sudo=outcome.action.requires_sudo,
                        approved=True,
                        executed=result.exit_code is not None or result.timed_out,
                        exit_code=result.exit_code,
                        duration_ms=result.duration_ms,
                        output_preview=(result.stdout + "\n" + result.stderr)[:8000],
                        note=result.error or outcome.reason,
                    )
                except HistoryError as exc:
                    console.print(f"[{WARN}]Не удалось сохранить действие: {exc}[/{WARN}]")

    if not planned.commands:
        return []
    validator = CommandValidator(
        allow_sudo=runtime.config.allow_sudo,
        cwd=Path.cwd(),
    )
    approval = ApprovalEngine(
        console,
        validator,
        allow_sudo=runtime.config.allow_sudo,
    )
    executor = Executor(runtime.config, validator=validator)
    executed: list[tuple[ValidatedCommand, ExecutionResult, str]] = []
    total = len(planned.commands)

    for index, raw in enumerate(planned.commands, start=1):
        try:
            validated = validator.validate(raw)
        except ValidationError as exc:
            _render_blocked(console, raw, exc.reason)
            with contextlib.suppress(HistoryError):
                runtime.history.add_command(
                    session.conversation_id,
                    argv=list(raw.argv),
                    description=raw.description,
                    risk=raw.risk,
                    requires_sudo=raw.requires_sudo,
                    approved=False,
                    executed=False,
                    note=f"заблокировано: {exc.reason}",
                )
            continue

        if not allow_exec:
            _render_planned(console, validated)
            continue

        outcome = approval.request_approval(validated, index=index, total=total)
        if not outcome.approved or outcome.command is None:
            with contextlib.suppress(HistoryError):
                runtime.history.add_command(
                    session.conversation_id,
                    argv=list(validated.argv),
                    description=validated.description,
                    risk=validated.risk,
                    requires_sudo=validated.requires_sudo,
                    approved=False,
                    executed=False,
                    note=outcome.reason,
                )
            continue

        result = executor.execute(outcome.command)
        result.stdout = redact_text(result.stdout, extra_secrets=runtime.secrets)
        result.stderr = redact_text(result.stderr, extra_secrets=runtime.secrets)
        if result.error:
            result.error = redact_text(result.error, extra_secrets=runtime.secrets)
        output = redact_text(result.combined, extra_secrets=runtime.secrets)
        _render_execution(console, outcome.command, result, output)
        if result.error and "sudo" in result.error.lower():
            console.print(
                f"[{WARN}]sudo -n не смог выполнить команду без пароля. "
                f"Выполните её вручную в терминале.[/{WARN}]"
            )
        try:
            runtime.history.add_command(
                session.conversation_id,
                argv=list(outcome.command.argv),
                description=outcome.command.description,
                risk=outcome.command.risk,
                requires_sudo=outcome.command.requires_sudo,
                approved=True,
                executed=True,
                exit_code=result.exit_code,
                duration_ms=result.duration_ms,
                output_preview=output[: 2 * runtime.config.max_output_chars + 1000],
                note=outcome.reason,
            )
        except HistoryError as exc:
            console.print(f"[{WARN}]Не удалось сохранить команду: {exc}[/{WARN}]")
        executed.append((outcome.command, result, output))
    return executed


def _analyze_results(
    runtime: Runtime,
    router: Router,
    session: ConversationSession,
    executed: list[tuple[ValidatedCommand, ExecutionResult, str]],
    *,
    provider: str | None,
    model: str | None,
) -> None:
    if not executed or not runtime.config.analyze_results:
        return
    chunks: list[str] = []
    budget = runtime.config.ai_output_chars
    for command, result, _output in executed:
        per_result_budget = max(1000, budget // max(1, len(executed)))
        structured = {
            "command": command.display(),
            "status": result.status,
            "exit_code": result.exit_code,
            "duration_ms": result.duration_ms,
            "summary": result.summary,
            "warnings": result.warnings,
            "inaccessible_count": result.inaccessible_count,
            "stdout": result.stdout[: per_result_budget // 2],
            "stderr": result.stderr[: per_result_budget // 2],
            "truncated_for_analysis": (
                len(result.stdout) > per_result_budget // 2
                or len(result.stderr) > per_result_budget // 2
            ),
        }
        chunk = json.dumps(structured, ensure_ascii=False, indent=2)
        chunks.append(chunk)
    analysis_text = "\n\n".join(chunks)[:budget]
    system_prompt = build_system_prompt(
        runtime.config,
        "Локальный контекст уже учтён в предыдущих сообщениях.",
    )
    request = ChatRequest(
        messages=[
            AIMessage(role="system", content=system_prompt),
            *session.history_window(),
            AIMessage(role="user", content=build_analysis_prompt(analysis_text)),
        ],
        temperature=runtime.config.temperature,
        max_tokens=runtime.config.max_tokens,
        json_mode=True,
    )
    response = _ask_ai(
        runtime.console,
        router,
        request,
        provider=provider,
        model=model,
    )
    if response is None:
        runtime.console.print(
            f"[{WARN}]Не удалось получить анализ результата, но вывод команды доступен выше.[/{WARN}]"
        )
        return
    action = parse_action(response.text)
    session.add_assistant(
        action.message or response.text,
        provider=response.provider,
        model=response.model,
    )
    _render_response(runtime.console, action.message or response.text, response)
    if action.commands or action.actions:
        runtime.console.print(
            f"[{WARN}]Планы из анализа не выполняются автоматически. "
            f"Сделайте отдельный запрос.[/{WARN}]"
        )


def _render_directory_inspection(console: Console, report: DirectoryInspection) -> None:
    is_partial = report.status == "partial_success"
    is_failed = report.status == "failed"
    title = "RYOKU AI — PARTIAL RESULT" if is_partial else (
        "RYOKU AI — INSPECTION FAILED" if is_failed else "RYOKU AI — DIRECTORY RESULT"
    )
    border = WARN if is_partial else DANGER if is_failed else ACCENT_SOFT
    body = Text(overflow="fold")
    body.append("Directory: ", style="bold")
    body.append(report.path + "\n")
    body.append("Readable size: ", style="bold")
    body.append(report.readable_size + "\n")
    body.append("Status: ", style="bold")
    body.append(f"{report.status.replace('_', ' ').title()}\n")
    body.append(f"Exit code: {report.exit_code}\n")
    body.append(f"Duration: {report.duration_ms} ms\n")
    body.append(f"Scanned entries: {report.scanned_entries}\n")
    body.append(f"Inaccessible directories: {report.inaccessible_directories}\n")
    body.append(f"Inaccessible entries: {report.inaccessible_entries}\n")
    body.append(f"Disappeared entries: {report.disappeared_entries}\n")
    body.append(f"Summary: {report.summary}\n", style=WARN if is_partial or is_failed else OK)
    for warning in report.warnings:
        body.append(f"Warning: {warning}\n", style=WARN)
    if report.readable_bytes == 0 and is_partial:
        body.append("No readable data was measured; zero does not mean the tree is empty.\n", style=WARN)
    console.print(Panel(body, title=Text(title, style=f"bold white on {border}"), border_style=border))


def _inspection_query_paths(query: str) -> list[Path] | None:
    lowered = query.lower()
    if "/var/tmp" in lowered or "var tmp" in lowered:
        return [Path("/var/tmp")]
    if "/tmp" in lowered or re.search(r"(?<!var)\btmp\b", lowered):
        return [Path("/tmp")]
    if ".cache" in lowered or (
        any(word in lowered for word in ("кеш", "кэш", "cache"))
        and any(word in lowered for word in ("мой", "моего", "пользователь", "домашн"))
    ):
        return [Path.home() / ".cache"]
    return None


def _handle_temp_inspection(
    runtime: Runtime,
    query: str,
    *,
    provider: str | None,
    model: str | None,
) -> bool:
    targets = _inspection_query_paths(query)
    if targets is None:
        return False
    session = ConversationSession(runtime.config, runtime.history, title=query, cwd=Path.cwd())
    session.add_user(query)
    reports = [inspect_directory(target) for target in targets]
    serialized = [report.as_dict() for report in reports]
    for report in reports:
        _render_directory_inspection(runtime.console, report)
        try:
            runtime.history.add_command(
                session.conversation_id,
                argv=["temp-inspect", report.path],
                description="Непривилегированный read-only обход каталога без перехода по symlink",
                risk="low",
                requires_sudo=False,
                approved=True,
                executed=True,
                exit_code=report.exit_code,
                duration_ms=report.duration_ms,
                output_preview=json.dumps(report.as_dict(), ensure_ascii=False),
                note=report.summary,
            )
        except HistoryError as exc:
            runtime.console.print(f"[{WARN}]Не удалось сохранить результат: {exc}[/{WARN}]")

    if not runtime.credentials.has_any():
        runtime.console.print(
            f"[{MUTED}]Локальный замер завершён. Для объяснения результата настройте AI: "
            f"ryoku-ai setup.[/{MUTED}]"
        )
        return True
    router = _build_router(runtime)
    request = ChatRequest(
        messages=[
            AIMessage(role="system", content=build_system_prompt(runtime.config, "")),
            AIMessage(
                role="user",
                content=build_analysis_prompt(
                    json.dumps(serialized, ensure_ascii=False, indent=2)
                ),
            ),
        ],
        temperature=runtime.config.temperature,
        max_tokens=runtime.config.max_tokens,
        json_mode=True,
    )
    response = _ask_ai(runtime.console, router, request, provider=provider, model=model)
    if response is not None:
        action = parse_action(response.text)
        session.add_assistant(action.message or response.text, provider=response.provider, model=response.model)
        _render_response(runtime.console, action.message or response.text, response)
    return True


def _explain_latest_command(
    runtime: Runtime,
    query: str,
    *,
    provider: str | None,
    model: str | None,
) -> bool:
    latest = runtime.history.latest_executed_command()
    if latest is None or not latest.get("output_preview"):
        _warn_panel(runtime.console, "Нет результата", "В истории пока нет сохранённого вывода команды.")
        return True
    stored_output = str(latest.get("output_preview") or "")
    stdout, separator, stderr = stored_output.partition("\n[stderr]\n")
    try:
        scanned_result = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        scanned_result = None
    if not runtime.credentials.has_any():
        if isinstance(scanned_result, dict) and "readable_bytes" in scanned_result:
            try:
                report = DirectoryInspection(
                    path=str(scanned_result["path"]),
                    readable_bytes=int(scanned_result["readable_bytes"]),
                    scanned_entries=int(scanned_result.get("scanned_entries", 0)),
                    inaccessible_directories=int(
                        scanned_result.get("inaccessible_directories", 0)
                    ),
                    inaccessible_entries=int(scanned_result.get("inaccessible_entries", 0)),
                    disappeared_entries=int(scanned_result.get("disappeared_entries", 0)),
                    duration_ms=int(scanned_result.get("duration_ms", 0)),
                    status=scanned_result.get("status", "failed"),
                    warnings=list(scanned_result.get("warnings", [])),
                    summary=str(scanned_result.get("summary", "")),
                )
                _render_directory_inspection(runtime.console, report)
            except (TypeError, ValueError):
                runtime.console.print(Text(stored_output, overflow="fold"))
        else:
            runtime.console.print(Text(stored_output, overflow="fold"))
        return True
    session = ConversationSession(runtime.config, runtime.history, title=query, cwd=Path.cwd())
    session.add_user(query)
    exit_code = latest.get("exit_code")
    has_partial_warnings = bool(
        re.search(
            r"permission denied|operation not permitted|access denied|no such file|cannot stat|vanished",
            stderr,
            re.I,
        )
    )
    result_status = (
        scanned_result.get("status")
        if isinstance(scanned_result, dict)
        and scanned_result.get("status") in {"success", "partial_success", "failed"}
        else
        "success"
        if exit_code == 0
        else "partial_success"
        if stdout.strip() and has_partial_warnings
        else "failed"
    )
    details = (
        {
            **scanned_result,
            "command": latest.get("argv", []),
            "exit_code": exit_code,
        }
        if isinstance(scanned_result, dict)
        else {
            "command": latest.get("argv", []),
            "status": result_status,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr if separator else "",
            "summary": latest.get("note"),
        }
    )
    router = _build_router(runtime)
    request = ChatRequest(
        messages=[
            AIMessage(role="system", content=build_system_prompt(runtime.config, "")),
            AIMessage(role="user", content=build_analysis_prompt(json.dumps(details, ensure_ascii=False))),
        ],
        temperature=runtime.config.temperature,
        max_tokens=runtime.config.max_tokens,
        json_mode=True,
    )
    response = _ask_ai(runtime.console, router, request, provider=provider, model=model)
    if response is not None:
        action = parse_action(response.text)
        session.add_assistant(action.message or response.text, provider=response.provider, model=response.model)
        _render_response(runtime.console, action.message or response.text, response)
    return True


def _execute_query(
    runtime: Runtime,
    query: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    allow_exec: bool = True,
) -> bool:
    console = runtime.console
    lowered_query = query.lower()
    if "последн" in lowered_query and ("результат" in lowered_query or "команд" in lowered_query):
        return _explain_latest_command(
            runtime,
            query,
            provider=provider,
            model=model,
        )
    if _handle_temp_inspection(runtime, query, provider=provider, model=model):
        return True
    router = _build_router(runtime)
    if not router.has_any_provider():
        _ai_error(console, NoProvidersConfigured("Не настроен ни один AI-провайдер."))
        return False

    session = ConversationSession(
        runtime.config,
        runtime.history,
        title=query,
        cwd=Path.cwd(),
    )
    session.add_user(query)
    context_text = collect_context() if runtime.config.auto_context else "Контекст отключён."
    system_prompt = build_system_prompt(runtime.config, context_text)
    request = session.build_request(system_prompt, model=model)
    response = _ask_ai(console, router, request, provider=provider, model=model)
    if response is None:
        return False

    action = parse_action(response.text)
    session.add_assistant(
        action.message or response.text,
        provider=response.provider,
        model=response.model,
    )
    _render_response(console, action.message or response.text, response)
    _display_action(console, action)
    executed = _run_commands(runtime, session, action, allow_exec=allow_exec)
    _analyze_results(
        runtime,
        router,
        session,
        executed,
        provider=provider,
        model=model,
    )
    with contextlib.suppress(HistoryError):
        runtime.history.prune(runtime.config.max_history)
    return True


def _run_explain(runtime: Runtime, command_text: str) -> bool:
    console = runtime.console
    router = _build_router(runtime)
    if not router.has_any_provider():
        _ai_error(console, NoProvidersConfigured("Не настроен ни один AI-провайдер."))
        return False
    context_text = collect_context() if runtime.config.auto_context else "Контекст отключён."
    system_prompt = build_system_prompt(runtime.config, context_text)
    session = ConversationSession(
        runtime.config,
        runtime.history,
        title=f"Объяснение: {command_text[:80]}",
        cwd=Path.cwd(),
    )
    user_text = build_explain_prompt(command_text)
    session.add_user(user_text)
    request = session.build_request(system_prompt)
    response = _ask_ai(console, router, request, provider=None, model=None)
    if response is None:
        return False
    action = parse_action(response.text)
    session.add_assistant(
        action.message or response.text,
        provider=response.provider,
        model=response.model,
    )
    _render_response(console, action.message or response.text, response)
    console.print(
        f"[{MUTED}]Команда была только объяснена и не выполнялась.[/{MUTED}]"
    )
    if action.commands or action.actions:
        console.print(
            f"[{WARN}]Модель предложила план, но в режиме объяснения он игнорируется.[/{WARN}]"
        )
    return True


def _chat(runtime: Runtime, *, provider: str | None = None, model: str | None = None) -> None:
    console = runtime.console
    router = _build_router(runtime)
    if not router.has_any_provider():
        _ai_error(console, NoProvidersConfigured("Не настроен ни один AI-провайдер."))
        return

    _banner(console)
    console.print(
        f"[{MUTED}]Чат. /help — команды, /exit — выход. "
        f"Провайдер: {runtime.config.provider}, модель: "
        f"{model or runtime.config.model or 'авто'}[/{MUTED}]"
    )
    context_text = collect_context() if runtime.config.auto_context else "Контекст отключён."
    system_prompt = build_system_prompt(runtime.config, context_text)
    session = ConversationSession(
        runtime.config,
        runtime.history,
        title="Интерактивный чат",
        cwd=Path.cwd(),
    )

    while True:
        try:
            line = console.input(f"[bold {ACCENT}]Вы › [/bold {ACCENT}]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n" + str(Text("Выход.", style=MUTED)))
            break
        if not line:
            continue
        if line in {"/exit", "/quit", "/q"}:
            console.print(Text("Выход.", style=MUTED))
            break
        if line == "/help":
            console.print(
                Panel(
                    Text(
                        "/help  — эта справка\n"
                        "/clear — очистить экран\n"
                        "/exit  — выйти\n\n"
                        "Любой текст — запрос к AI. Команды выполняются только после "
                        "отдельного подтверждения.",
                    ),
                    border_style=ACCENT_SOFT,
                )
            )
            continue
        if line == "/clear":
            console.clear()
            _banner(console)
            continue

        session.add_user(line)
        request = session.build_request(system_prompt, model=model)
        response = _ask_ai(console, router, request, provider=provider, model=model)
        if response is None:
            continue
        action = parse_action(response.text)
        session.add_assistant(
            action.message or response.text,
            provider=response.provider,
            model=response.model,
        )
        _render_response(console, action.message or response.text, response)
        _display_action(console, action)
        executed = _run_commands(runtime, session, action)
        _analyze_results(
            runtime,
            router,
            session,
            executed,
            provider=provider,
            model=model,
        )
        with contextlib.suppress(HistoryError):
            runtime.history.prune(runtime.config.max_history)


# --- Диагностика -----------------------------------------------------------


def _record_diagnostic(runtime: Runtime, kind: str, summary: str, payload: Any) -> None:
    try:
        runtime.history.add_diagnostic(kind, summary, payload)
    except HistoryError as exc:
        runtime.console.print(f"[{WARN}]История: {exc}[/{WARN}]")


def _render_network(console: Console, report: NetworkReport) -> None:
    status = Text("интернет доступен", style=f"bold {OK}") if report.internet_ok else Text(
        "интернет недоступен", style=f"bold {DANGER}"
    )
    header = Text()
    header.append("СЕТЬ\n", style=f"bold {ACCENT}")
    header.append("Шлюз: ")
    header.append(report.default_route + "\n")
    header.append("DNS: ")
    header.append(", ".join(report.dns_servers) if report.dns_servers else "не найден")
    header.append("\nСтатус: ")
    header.append_text(status)
    console.print(Panel(header, border_style=ACCENT_SOFT, padding=(0, 2)))

    interfaces = Table(title="Интерфейсы", box=box.SIMPLE_HEAD, header_style=f"bold {ACCENT}")
    interfaces.add_column("Интерфейс")
    interfaces.add_column("Состояние")
    interfaces.add_column("Адреса")
    interfaces.add_column("Всего ↓/↑", justify="right")
    interfaces.add_column("Сейчас ↓/↑", justify="right")
    for item in report.interfaces:
        interfaces.add_row(
            Text(item.name, style="bold"),
            Text(item.state),
            Text("\n".join(item.addresses) if item.addresses else "—"),
            Text(f"{format_bytes(item.rx_bytes)} / {format_bytes(item.tx_bytes)}"),
            Text(f"{format_bytes(item.rx_rate)}/s / {format_bytes(item.tx_rate)}/s"),
        )
    console.print(interfaces)

    if report.internet_checks:
        checks = Table(title="Проверка доступности", box=box.SIMPLE, show_header=False)
        checks.add_column("Проверка")
        checks.add_column("Результат")
        for check in report.internet_checks:
            checks.add_row(
                Text(check.name),
                Text(("✓ " if check.ok else "✗ ") + check.detail, style=OK if check.ok else DANGER),
            )
        console.print(checks)

    if report.state_counts:
        summary = "  ".join(
            f"{state}: {count}" for state, count in report.state_counts.items()
        )
        console.print(f"\n[{MUTED}]Состояния соединений: {summary}[/{MUTED}]")

    if report.connections:
        connections = Table(
            title=f"Соединения (показано {min(20, len(report.connections))} из {len(report.connections)})",
            box=box.SIMPLE_HEAD,
            header_style=f"bold {ACCENT}",
        )
        connections.add_column("Протокол")
        connections.add_column("Состояние")
        connections.add_column("Локальный адрес")
        connections.add_column("Удалённый адрес")
        connections.add_column("Процесс")
        for conn in report.connections[:20]:
            connections.add_row(
                Text(conn.netid),
                Text(conn.state),
                Text(conn.local),
                Text(conn.peer),
                Text(conn.process or "—"),
            )
        console.print(connections)

    if report.top_talkers:
        talkers = Table(title="Процессы с сетевой активностью", box=box.SIMPLE)
        talkers.add_column("Процесс")
        talkers.add_column("Сокетов", justify="right")
        for name, count in report.top_talkers:
            talkers.add_row(Text(name), Text(str(count)))
        console.print(talkers)

    if report.limited_process_info:
        console.print(
            f"[{WARN}]Информация о процессах ограничена: ss запущен без root.[/{WARN}]"
        )
    for error in report.errors:
        console.print(f"[{WARN}]⚠ {error}[/{WARN}]")


def _render_system(console: Console, report: SystemReport) -> None:
    header = Text()
    header.append(report.os_name, style=f"bold {ACCENT}")
    header.append(f"  ·  ядро {report.kernel}  ·  {report.arch}  ·  uptime {report.uptime_text}")
    console.print(Panel(header, border_style=ACCENT_SOFT, padding=(0, 2)))

    cpu = report.cpu
    cpu_line = Text()
    cpu_line.append("CPU: ", style="bold")
    cpu_line.append(f"{cpu.model}\n")
    cpu_line.append(
        f"Ядра: {cpu.cores}  ·  загрузка {cpu.usage_percent}%  ·  "
        f"load {cpu.load1:.2f} / {cpu.load5:.2f} / {cpu.load15:.2f}"
    )
    mem = report.memory
    mem_line = Text()
    mem_line.append("\nRAM: ", style="bold")
    mem_line.append(
        f"{mem.total_text} всего, доступно {mem.available_text} "
        f"({mem.used_percent}% занято)\n"
    )
    mem_line.append("Swap: ", style="bold")
    mem_line.append(
        f"{mem.swap_used_text} из {mem.swap_total_text}"
        if mem.swap_total_kb
        else "не используется"
    )
    mem_line.append(f"\nПроцессов: {report.process_count}  ·  зомби: {report.zombie_count}")
    console.print(Panel(cpu_line + mem_line, title="Ресурсы", border_style=ACCENT_SOFT))

    if report.disks:
        disks = Table(title="Диски", box=box.SIMPLE_HEAD, header_style=f"bold {ACCENT}")
        disks.add_column("Устройство")
        disks.add_column("Точка")
        disks.add_column("ФС")
        disks.add_column("Всего", justify="right")
        disks.add_column("Занято", justify="right")
        disks.add_column("Свободно", justify="right")
        disks.add_column("%", justify="right")
        for disk in report.disks:
            percent_style = DANGER if disk.percent >= 90 else WARN if disk.percent >= 75 else OK
            disks.add_row(
                Text(disk.device),
                Text(disk.mountpoint, style="bold"),
                Text(disk.fstype),
                Text(disk.total_text),
                Text(disk.used_text),
                Text(disk.free_text),
                Text(f"{disk.percent}%", style=percent_style),
            )
        console.print(disks)

    if report.btrfs_usage or report.btrfs_error:
        console.print(
            Panel(
                Syntax(report.btrfs_usage, "text", theme="monokai", word_wrap=True)
                if report.btrfs_usage
                else Text(report.btrfs_error or "", style=WARN),
                title="Btrfs",
                border_style=ACCENT_SOFT,
            )
        )

    if report.temperatures:
        temps = Table(title="Температуры", box=box.SIMPLE_HEAD, header_style=f"bold {ACCENT}")
        temps.add_column("Источник")
        temps.add_column("Датчик")
        temps.add_column("°C", justify="right")
        for item in report.temperatures:
            style = DANGER if item.celsius >= 90 else WARN if item.celsius >= 75 else OK
            temps.add_row(Text(item.source), Text(item.label), Text(f"{item.celsius:.1f}", style=style))
        console.print(temps)

    if report.gpus:
        gpus = Table(title="GPU", box=box.SIMPLE_HEAD, header_style=f"bold {ACCENT}")
        gpus.add_column("Модель")
        gpus.add_column("VRAM", justify="right")
        gpus.add_column("Загрузка", justify="right")
        gpus.add_column("Темп.", justify="right")
        for item in report.gpus:
            memory = (
                f"{item.memory_used_mb}/{item.memory_total_mb} MiB"
                if item.memory_used_mb is not None and item.memory_total_mb is not None
                else "—"
            )
            gpus.add_row(
                Text(item.name),
                Text(memory),
                Text(f"{item.utilization}%" if item.utilization is not None else "—"),
                Text(f"{item.temperature}°C" if item.temperature is not None else "—"),
            )
        console.print(gpus)

    for error in report.errors:
        console.print(f"[{WARN}]⚠ {error}[/{WARN}]")


def _render_processes(console: Console, report: ProcessesReport) -> None:
    console.print(
        Panel(
            Text(
                f"Всего процессов: {report.total}  ·  зомби: {len(report.zombies)}",
                style=f"bold {ACCENT}",
            ),
            border_style=ACCENT_SOFT,
        )
    )
    for title, processes in (
        ("Топ по CPU", report.top_cpu),
        ("Топ по памяти", report.top_mem),
    ):
        table = Table(title=title, box=box.SIMPLE_HEAD, header_style=f"bold {ACCENT}")
        table.add_column("PID", justify="right")
        table.add_column("Пользователь")
        table.add_column("CPU %", justify="right")
        table.add_column("MEM %", justify="right")
        table.add_column("Время")
        table.add_column("Команда")
        for item in processes:
            table.add_row(
                Text(str(item.pid)),
                Text(item.user),
                Text(f"{item.cpu:.1f}"),
                Text(f"{item.mem:.1f}"),
                Text(item.elapsed),
                Text(item.command),
            )
        console.print(table)

    if report.zombies:
        table = Table(title="Зомби-процессы", box=box.SIMPLE_HEAD, header_style=f"bold {DANGER}")
        table.add_column("PID")
        table.add_column("Пользователь")
        table.add_column("Команда")
        for item in report.zombies:
            table.add_row(Text(str(item.pid)), Text(item.user), Text(item.command))
        console.print(table)

    if report.network_active:
        table = Table(title="Сетевая активность процессов", box=box.SIMPLE_HEAD)
        table.add_column("Процесс")
        table.add_column("Сокетов", justify="right")
        for name, count in report.network_active:
            table.add_row(Text(name), Text(str(count)))
        console.print(table)
    if report.limited_network_info:
        console.print(f"[{WARN}]Процессы в ss скрыты без root.[/{WARN}]")
    for error in report.errors:
        console.print(f"[{WARN}]⚠ {error}[/{WARN}]")


def _render_storage(console: Console, report: StorageReport) -> None:
    if report.disks:
        table = Table(title="Файловые системы", box=box.SIMPLE_HEAD, header_style=f"bold {ACCENT}")
        table.add_column("Устройство")
        table.add_column("Точка")
        table.add_column("ФС")
        table.add_column("Всего", justify="right")
        table.add_column("Свободно", justify="right")
        table.add_column("%", justify="right")
        for disk in report.disks:
            style = DANGER if disk.percent >= 90 else WARN if disk.percent >= 75 else OK
            table.add_row(
                Text(disk.device),
                Text(disk.mountpoint, style="bold"),
                Text(disk.fstype),
                Text(disk.total_text),
                Text(disk.free_text),
                Text(f"{disk.percent}%", style=style),
            )
        console.print(table)
    if report.inodes:
        table = Table(title="Inode", box=box.SIMPLE)
        table.add_column("Точка")
        table.add_column("Всего", justify="right")
        table.add_column("Свободно", justify="right")
        table.add_column("%", justify="right")
        for item in report.inodes:
            table.add_row(
                Text(item.mountpoint),
                Text(str(item.total)),
                Text(str(item.free)),
                Text(f"{item.used_percent}%"),
            )
        console.print(table)
    if report.btrfs_usage:
        console.print(
            Panel(
                Syntax(report.btrfs_usage, "text", theme="monokai", word_wrap=True),
                title="Btrfs",
                border_style=ACCENT_SOFT,
            )
        )
    elif report.btrfs_error:
        console.print(f"[{WARN}]Btrfs: {report.btrfs_error}[/{WARN}]")
    if report.top_dirs:
        table = Table(title="Крупнейшие каталоги", box=box.SIMPLE_HEAD)
        table.add_column("Размер", justify="right")
        table.add_column("Каталог")
        for path, size in report.top_dirs:
            table.add_row(Text(format_bytes(size)), Text(path))
        console.print(table)
    for error in report.errors:
        console.print(f"[{WARN}]⚠ {error}[/{WARN}]")


def _render_services(console: Console, report: ServiceReport, *, docker_only: bool = False) -> None:
    if not docker_only:
        state_style = OK if report.system_state == "running" else WARN
        console.print(
            Panel(
                Text(
                    f"systemd: {report.system_state}  ·  "
                    f"запущено служб: {report.running_services}  ·  "
                    f"failed: {len(report.failed_units)}",
                    style=state_style,
                ),
                title="Службы",
                border_style=ACCENT_SOFT,
            )
        )
        if report.failed_units:
            table = Table(title="Неудачные unit", box=box.SIMPLE_HEAD, header_style=f"bold {DANGER}")
            table.add_column("Unit")
            table.add_column("Состояние")
            table.add_column("Описание")
            for unit in report.failed_units:
                table.add_row(
                    Text(unit.name, style="bold"),
                    Text(f"{unit.active}/{unit.sub}"),
                    Text(unit.description),
                )
            console.print(table)
        if report.journal_errors:
            console.print(
                Panel(
                    Syntax(
                        "\n".join(report.journal_errors[-10:]),
                        "text",
                        theme="monokai",
                        word_wrap=True,
                    ),
                    title="Ошибки журнала (priority 3, текущая загрузка)",
                    border_style=WARN,
                )
            )

    _render_docker(console, report.docker)
    if report.steam_running:
        console.print(
            Panel(
                Text("\n".join(report.steam_processes), style=OK),
                title="Steam запущен",
                border_style=OK,
            )
        )
    else:
        console.print(f"[{MUTED}]Steam: не запущен[/{MUTED}]")
    for error in report.errors:
        if docker_only and error.startswith("steam:"):
            continue
        console.print(f"[{WARN}]⚠ {error}[/{WARN}]")


def _render_docker(console: Console, info: DockerInfo) -> None:
    if not info.installed:
        console.print(
            Panel(
                Text(
                    "Docker не установлен. Установка выполняется вручную:\n"
                    "  sudo pacman -S docker docker-compose\n"
                    "  sudo systemctl enable --now docker\n\n"
                    "Ryoku AI Terminal не устанавливает пакеты автоматически.",
                    style=MUTED,
                ),
                title="Docker",
                border_style=WARN,
            )
        )
        return
    header = Text()
    header.append(f"Служба: {info.service_state}\n", style="bold")
    header.append(f"Версия: {info.version or 'неизвестно'}")
    if info.error:
        header.append(f"\n{info.error}", style=DANGER)
    console.print(Panel(header, title="Docker", border_style=ACCENT_SOFT))
    if info.containers:
        table = Table(title="Контейнеры", box=box.SIMPLE_HEAD, header_style=f"bold {ACCENT}")
        table.add_column("Имя")
        table.add_column("Статус")
        table.add_column("Образ")
        for line in info.containers:
            parts = line.split("\t")
            parts += [""] * (3 - len(parts))
            table.add_row(Text(parts[0]), Text(parts[1]), Text(parts[2]))
        console.print(table)
    else:
        console.print(f"[{MUTED}]Контейнеров нет или Docker недоступен.[/{MUTED}]")
    if info.disk_usage:
        console.print(
            Panel(
                Syntax(info.disk_usage, "text", theme="monokai", word_wrap=True),
                title="docker system df",
                border_style=ACCENT_SOFT,
            )
        )


# --- История ---------------------------------------------------------------


def _format_time(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value[:16]


def _show_history(runtime: Runtime, *, limit: int = 20, show_id: int | None = None) -> None:
    console = runtime.console
    if show_id is not None:
        conversation = runtime.history.get_conversation(show_id)
        if conversation is None:
            _error_panel(console, "История", f"Диалог #{show_id} не найден.")
            return
        console.print(
            Panel(
                Text(
                    f"#{conversation.id}  ·  {_format_time(conversation.started_at)}  ·  "
                    f"{conversation.provider or '—'} / {conversation.model or '—'}\n"
                    f"{conversation.title or 'Без названия'}"
                ),
                title="Диалог",
                border_style=ACCENT_SOFT,
            )
        )
        messages = runtime.history.list_messages(show_id)
        for message in messages:
            color = ACCENT if message.role == "assistant" else "white"
            if message.role == "system":
                color = MUTED
            console.print(
                Panel(
                    Text(message.content),
                    title=Text(
                        f"{message.role} · {_format_time(message.created_at)}",
                        style=color,
                    ),
                    border_style=color,
                    padding=(0, 1),
                )
            )
        command_results = [
            item
            for item in runtime.history.list_commands(100)
            if item.get("conversation_id") == show_id and item.get("output_preview")
        ]
        for item in reversed(command_results):
            command_text = " ".join(str(part) for part in item.get("argv", []))
            output = str(item.get("output_preview") or "")
            console.print(
                Panel(
                    Text(output, overflow="fold"),
                    title=Text(
                        f"Полный вывод · {command_text} · exit {item.get('exit_code')}"
                    ),
                    border_style=MUTED,
                )
            )
        return

    conversations = runtime.history.list_conversations(limit)
    if not conversations:
        console.print(f"[{MUTED}]История пуста.[/{MUTED}]")
        return
    table = Table(
        title=f"История диалогов (последние {len(conversations)})",
        box=box.SIMPLE_HEAD,
        header_style=f"bold {ACCENT}",
    )
    table.add_column("ID", justify="right")
    table.add_column("Когда")
    table.add_column("Сообщений", justify="right")
    table.add_column("Модель")
    table.add_column("Тема")
    for item in conversations:
        table.add_row(
            Text(str(item.id)),
            Text(_format_time(item.updated_at)),
            Text(str(item.message_count)),
            Text(f"{item.provider or '—'} / {item.model or '—'}"),
            Text(item.title or "—"),
        )
    console.print(table)
    console.print(
        f"[{MUTED}]Просмотр: ai --history-show ID  ·  очистка: ai --history-clear[/{MUTED}]"
    )

    commands = runtime.history.list_commands(10)
    if commands:
        cmd_table = Table(title="Последние команды", box=box.SIMPLE)
        cmd_table.add_column("Когда")
        cmd_table.add_column("Команда")
        cmd_table.add_column("Итог")
        cmd_table.add_column("Код", justify="right")
        for item in commands:
            argv = item.get("argv") or []
            command_text = " ".join(str(part) for part in argv)
            result = "выполнена" if item.get("executed") else (
                "подтверждена" if item.get("approved") else "не выполнена"
            )
            cmd_table.add_row(
                Text(_format_time(str(item.get("created_at", "")))),
                Text(command_text),
                Text(result),
                Text(str(item.get("exit_code") if item.get("exit_code") is not None else "—")),
            )
        console.print(cmd_table)


def _clear_history(runtime: Runtime, *, assume_yes: bool = False) -> None:
    console = runtime.console
    if not runtime.config.history_enabled:
        console.print(f"[{MUTED}]История отключена в конфигурации.[/{MUTED}]")
        return
    if not assume_yes:
        console.print(
            f"[{WARN}]Будут удалены все диалоги, команды и результаты диагностики.[/{WARN}]"
        )
        try:
            confirmed = typer.confirm("Удалить историю?", default=False)
        except (EOFError, typer.Abort):
            confirmed = False
        if not confirmed:
            console.print(f"[{MUTED}]Отменено.[/{MUTED}]")
            return
    try:
        runtime.history.clear()
        console.print(f"[{OK}]История очищена.[/{OK}]")
    except HistoryError as exc:
        _error_panel(console, "История", str(exc))


# --- Модели ----------------------------------------------------------------


def _show_models(runtime: Runtime, *, provider: str | None = None) -> None:
    console = runtime.console
    router = _build_router(runtime)
    names = [provider] if provider else list(PROVIDER_NAMES)
    table = Table(title="Доступные модели", box=box.SIMPLE_HEAD, header_style=f"bold {ACCENT}")
    table.add_column("Провайдер")
    table.add_column("Модель")
    table.add_column("Контекст", justify="right")
    table.add_column("Бесплатно")
    table.add_column("Выбор")
    table.add_column("Статус")
    errors: list[tuple[str, str]] = []
    for name in names:
        if name not in router.providers:
            status = "нет API-ключа" if name in PROVIDER_NAMES else "неизвестный провайдер"
            table.add_row(
                Text(name), Text("—"), Text("—"), Text("—"), Text("—"),
                Text(status, style=MUTED),
            )
            continue
        with console.status(f"[{ACCENT}]Загрузка моделей {name}…[/{ACCENT}]", spinner="dots"):
            try:
                models = asyncio.run(router.list_models(name))
            except AIError as exc:
                message = redact_text(str(exc), extra_secrets=runtime.secrets)
                errors.append((name, message))
                table.add_row(
                    Text(name), Text("—"), Text("—"), Text("—"), Text("—"),
                    Text("ошибка списка", style=WARN),
                )
                continue
            except Exception as exc:  # noqa: BLE001
                message = redact_text(
                    f"{type(exc).__name__}: {exc}", extra_secrets=runtime.secrets
                )
                errors.append((name, message))
                table.add_row(
                    Text(name), Text("—"), Text("—"), Text("—"), Text("—"),
                    Text("ошибка списка", style=WARN),
                )
                continue
        selected = runtime.config.provider_models.get(name)
        models = [
            item for item in models if router.providers[name]._is_chat_model(item.id)
        ]
        if name == "openrouter":
            models = [item for item in models if item.id.endswith(":free")]
        for item in models[:40]:
            table.add_row(
                Text(name),
                Text(item.id),
                Text(str(item.context_length) if item.context_length else "—", justify="right"),
                Text("да" if item.free else "—" if item.free is None else "нет"),
                Text("← выбрана" if selected == item.id else "", style=OK),
                Text("доступна", style=OK),
            )
        if not models:
            table.add_row(Text(name), Text("—"), Text("—"), Text("—"), Text("—"), Text("моделей нет", style=WARN))
    console.print(table)
    for error_provider, message in errors:
        console.print(f"[{WARN}]⚠ {error_provider}: {message}[/{WARN}]")
    if not errors and router.has_any_provider():
        console.print(
            f"[{MUTED}]Выбрать модель: ai --provider groq --model <id> \"запрос\" "
            f"или укажите provider_models в config.toml[/{MUTED}]"
        )


# --- Конфигурация ----------------------------------------------------------


def _show_config(runtime: Runtime) -> None:
    console = runtime.console
    config = runtime.config
    table = Table(title="Конфигурация", box=box.SIMPLE_HEAD, header_style=f"bold {ACCENT}")
    table.add_column("Параметр")
    table.add_column("Значение")
    rows = [
        ("Язык", config.language),
        ("Тема", config.theme),
        ("Провайдер", config.provider),
        ("Порядок", " → ".join(config.provider_order)),
        ("Модель (общая)", config.model or "авто"),
        ("Резервные провайдеры", "включены" if config.fallback_enabled else "выключены"),
        ("Модели OpenRouter", "только с пометкой :free"),
        ("Таймаут запроса", f"{config.request_timeout:.0f} с"),
        ("Таймаут команды", f"{config.command_timeout:.0f} с"),
        ("Лимит вывода", f"{config.max_output_chars} символов"),
        ("Подтверждение команд", "всегда включено"),
        ("sudo", "разрешён с отдельным подтверждением" if config.allow_sudo else "запрещён"),
        ("История", "включена" if config.history_enabled else "выключена"),
        ("Контекст системы", "да" if config.auto_context else "нет"),
        ("Анализ результатов", "да" if config.analyze_results else "нет"),
        ("Лимит диалогов", str(config.max_history)),
    ]
    for name, value in rows:
        table.add_row(Text(name), Text(value))
    console.print(table)

    console.print(_provider_table(runtime))

    paths = Table(box=box.SIMPLE, show_header=False)
    paths.add_column("", style=MUTED)
    paths.add_column("")
    for label, path in (
        ("Конфигурация", config_path()),
        ("Ключи", credentials_path()),
        ("История", history_db_path()),
        ("Состояние", state_dir()),
    ):
        mode = ""
        with contextlib.suppress(OSError):
            mode = f"  ({path.stat().st_mode & 0o777:o})" if path.exists() else "  (нет)"
        paths.add_row(label, Text(str(path) + mode))
    console.print(paths)
    stats = runtime.history.stats()
    console.print(
        f"[{MUTED}]История: {stats['conversations']} диалогов, "
        f"{stats['messages']} сообщений, {stats['commands']} команд, "
        f"{stats['diagnostics']} диагностик[/{MUTED}]"
    )


def _init_config() -> None:
    console = _bootstrap_console()
    path = config_path()
    if path.exists():
        console.print(f"[{MUTED}]Конфигурация уже существует: {path}[/{MUTED}]")
        return
    saved = save_config(Config())
    console.print(f"[{OK}]Создана конфигурация: {saved}[/{OK}]")


# --- Установка ключей ------------------------------------------------------


_PROVIDER_INFO = {
    "groq": (
        "Groq",
        "https://console.groq.com/keys",
        "Быстрые бесплатные модели, ограниченные квоты.",
    ),
    "openrouter": (
        "OpenRouter",
        "https://openrouter.ai/keys",
        "Модели с суффиксом :free; лимиты меняются.",
    ),
}


def _validate_provider_key(
    provider: str,
    key: str,
    config: Config,
) -> tuple[bool, str]:
    cls = PROVIDER_CLASSES[provider]
    timeout = httpx.Timeout(20.0, connect=10.0)
    instance = cls(key, timeout=timeout)
    try:
        models = asyncio.run(instance.list_models())
    except AIError as exc:
        return False, redact_text(str(exc), extra_secrets=[key])
    except Exception as exc:  # noqa: BLE001
        return False, redact_text(f"{type(exc).__name__}: {exc}", extra_secrets=[key])
    if not models:
        return False, "ключ принят, но список моделей пуст"
    return True, f"доступно моделей: {len(models)}"


def _setup(runtime: Runtime) -> None:
    console = runtime.console
    _banner(console)
    console.print(
        Panel(
            Text(
                "Настроим бесплатные AI-провайдеры. Ключи сохраняются только локально\n"
                "в ~/.config/ryoku-ai/credentials.env с правами 600 и никогда не\n"
                "попадают в историю или логи.\n\n"
                "Бесплатные тарифы ограничены квотами. Платная подписка не требуется.",
            ),
            title="Первоначальная настройка",
            border_style=ACCENT_SOFT,
        )
    )
    info = Table(box=box.SIMPLE_HEAD, header_style=f"bold {ACCENT}")
    info.add_column("Провайдер")
    info.add_column("Ключ")
    info.add_column("Где получить")
    info.add_column("Примечание")
    for name in PROVIDER_NAMES:
        title, url, note = _PROVIDER_INFO[name]
        configured = "уже настроен" if getattr(runtime.credentials, name) else "нет"
        info.add_row(Text(title), Text(configured), Text(url), Text(note))
    console.print(info)

    updated = runtime.credentials.model_copy(deep=True)
    for name in PROVIDER_NAMES:
        title, url, _note = _PROVIDER_INFO[name]
        existing = getattr(updated, name)
        if existing:
            try:
                replace = typer.confirm(f"Обновить ключ {title}?", default=False)
            except (EOFError, typer.Abort):
                replace = False
            if not replace:
                continue
        console.print(f"\n[bold]{title}[/bold] — {url}")
        try:
            key = Prompt.ask(
                "API-ключ (Enter — пропустить)",
                default="",
                password=True,
                console=console,
            ).strip()
        except (EOFError, KeyboardInterrupt):
            key = ""
        if not key:
            continue
        if not validate_secret_format(key):
            _warn_panel(console, "Ключ", "Ключ пустой или содержит недопустимые символы.")
            continue
        with console.status(f"[{ACCENT}]Проверка ключа {title}…[/{ACCENT}]", spinner="dots"):
            ok, detail = _validate_provider_key(name, key, runtime.config)
        if ok:
            setattr(updated, name, key)
            save_credentials(updated)
            console.print(f"[{OK}]Ключ {title} сохранён. {detail}[/{OK}]")
        else:
            _warn_panel(
                console,
                f"{title}: проверка не удалась",
                f"{detail}\n\nКлюч не сохранён. Проверьте интернет, ключ и квоты.",
            )

    configured = updated.configured_providers()
    if not configured:
        console.print(
            f"\n[{WARN}]Ни один ключ не сохранён. Настройку можно повторить: "
            f"ryoku-ai setup[/{WARN}]"
        )
        return

    try:
        default_provider = typer.prompt(
            "Провайдер по умолчанию (auto/" + "/".join(configured) + ")",
            default="auto",
        ).strip()
    except (EOFError, typer.Abort):
        default_provider = "auto"
    if default_provider not in {"auto", *configured}:
        default_provider = "auto"
    runtime.config.provider = default_provider  # type: ignore[assignment]

    if default_provider != "auto":
        try:
            model = typer.prompt(
                f"Модель для {default_provider} (Enter — авто)",
                default="",
                show_default=False,
            ).strip()
        except (EOFError, typer.Abort):
            model = ""
        if model:
            runtime.config.provider_models[default_provider] = model
        else:
            runtime.config.provider_models.pop(default_provider, None)

    runtime.config.model = None
    saved_config = save_config(runtime.config)
    saved_creds = save_credentials(updated)
    console.print(
        Panel(
            Text(
                f"Готово.\n\n"
                f"Конфигурация: {saved_config}\n"
                f"Ключи: {saved_creds} (chmod 600)\n\n"
                f"Проверка:  ai --version\n"
                f"Модели:    ai --models\n"
                f"Запрос:    ai почему медленный интернет\n"
                f"Чат:       ai --chat\n\n"
                f"Ключи не сохраняются в истории. Для ротации удалите файл ключей "
                f"или запустите setup снова.",
                style=OK,
            ),
            title="Настройка завершена",
            border_style=OK,
        )
    )


# --- Doctor ----------------------------------------------------------------


def _doctor(runtime: Runtime) -> None:
    console = runtime.console
    table = Table(title="Проверка окружения", box=box.SIMPLE_HEAD, header_style=f"bold {ACCENT}")
    table.add_column("Компонент")
    table.add_column("Статус")
    table.add_column("Детали")

    def add(name: str, ok: bool, detail: str = "", *, optional: bool = False) -> None:
        style = OK if ok else (MUTED if optional else WARN)
        table.add_row(Text(name), Text("✓" if ok else "—", style=style), Text(detail))

    os_name = Path("/etc/os-release").exists()
    add("Linux /etc/os-release", os_name, "Arch-совместимая система ожидается")
    add("Python", sys.version_info >= (3, 10), sys.version.split()[0])
    add("Fish", bool(find_tool("fish")), find_tool("fish") or "не найден", optional=True)
    # Launcher живёт в ~/.local/bin, поэтому для него нужен обычный PATH,
    # а не ограниченный SAFE_PATH диагностических утилит.
    app = shutil.which("ryoku-ai")
    add("ryoku-ai", bool(app), app or "не найден в PATH")
    ai_function = Path.home() / ".config/fish/functions/ai.fish"
    add("Fish-функция ai", ai_function.exists(), str(ai_function), optional=True)
    for tool, package, optional in (
        ("ip", "iproute2", False),
        ("ss", "iproute2", False),
        ("ps", "procps-ng", False),
        ("df", "coreutils", False),
        ("dig", "bind", True),
        ("journalctl", "systemd", False),
        ("btrfs", "btrfs-progs", True),
        ("sensors", "lm_sensors", True),
        ("docker", "docker", True),
        ("nvidia-smi", "nvidia-utils", True),
    ):
        path = find_tool(tool)
        add(tool, bool(path), path or f"установите {package}", optional=optional)

    config_ok = config_path().exists()
    add("config.toml", config_ok, str(config_path()), optional=not config_ok)
    creds_ok = credentials_path().exists()
    add("credentials.env", creds_ok, str(credentials_path()), optional=not creds_ok)
    issue = credentials_permission_issue()
    add("Права ключей 600", issue is None, issue or "ok", optional=not creds_ok)
    add("Провайдеры", runtime.credentials.has_any(), ", ".join(runtime.credentials.configured_providers()) or "нет ключей")
    console.print(table)


# --- Typer -----------------------------------------------------------------


app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
    help=APP_HELP,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Подкоманды вынесены в отдельное приложение: variadic-аргумент query в
# основном приложении иначе съедает имя подкоманды. main() явно направляет
# первый аргумент сюда.
command_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

_SUBCOMMANDS = frozenset({"setup", "models", "history", "config", "doctor"})


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    query: list[str] = typer.Argument(  # noqa: B008
        None, help="Запрос к AI на русском языке."
    ),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Показать версию.",
    ),
    chat: bool = typer.Option(False, "--chat", help="Интерактивный чат."),
    explain: str | None = typer.Option(None, "--explain", metavar="КОМАНДА", help="Объяснить команду."),
    network: bool = typer.Option(False, "--network", help="Диагностика сети."),
    system: bool = typer.Option(False, "--system", help="Диагностика системы."),
    processes: bool = typer.Option(False, "--processes", help="Диагностика процессов."),
    docker: bool = typer.Option(False, "--docker", help="Состояние Docker и Steam."),
    storage: bool = typer.Option(False, "--storage", help="Диски, Btrfs и крупные каталоги."),
    services: bool = typer.Option(False, "--services", help="systemd и журнал ошибок."),
    history_flag: bool = typer.Option(False, "--history", help="Показать историю."),
    history_clear: bool = typer.Option(False, "--history-clear", help="Очистить историю."),
    history_show: int | None = typer.Option(None, "--history-show", metavar="ID", help="Показать диалог."),
    models: bool = typer.Option(False, "--models", help="Показать доступные модели."),
    config_flag: bool = typer.Option(False, "--config", help="Показать конфигурацию."),
    provider: str | None = typer.Option(None, "--provider", help="Провайдер: groq/openrouter."),
    model: str | None = typer.Option(None, "--model", help="Модель для запроса."),
    no_exec: bool = typer.Option(False, "--no-exec", help="Только показать план, не выполнять."),
) -> None:
    """Ryoku AI Terminal."""
    if ctx.invoked_subcommand is not None:
        return
    if version:
        return

    runtime = _load_runtime()
    console = runtime.console

    try:
        if history_clear:
            _clear_history(runtime)
            if not any((network, system, processes, docker, storage, services, models, config_flag, explain, chat, query)):
                return
        if history_show is not None:
            _show_history(runtime, show_id=history_show)
            return
        if history_flag:
            _show_history(runtime)
            return
        if models:
            _show_models(runtime, provider=provider)
            return
        if config_flag:
            _show_config(runtime)
            return
        if explain:
            _run_explain(runtime, explain)
            return

        did_diagnostic = False
        if network:
            did_diagnostic = True
            with console.status(f"[{ACCENT}]Сбор сетевой диагностики…[/{ACCENT}]", spinner="dots"):
                report = collect_network()
            _render_network(console, report)
            _record_diagnostic(
                runtime,
                "network",
                f"Интернет: {'да' if report.internet_ok else 'нет'}; "
                f"интерфейсов: {len(report.interfaces)}; соединений: {len(report.connections)}",
                asdict(report),
            )
        if system:
            did_diagnostic = True
            with console.status(f"[{ACCENT}]Сбор системной диагностики…[/{ACCENT}]", spinner="dots"):
                report = collect_system()
            _render_system(console, report)
            _record_diagnostic(
                runtime,
                "system",
                f"CPU {report.cpu.usage_percent}%, RAM {report.memory.used_percent}%, "
                f"дисков: {len(report.disks)}",
                asdict(report),
            )
        if processes:
            did_diagnostic = True
            with console.status(f"[{ACCENT}]Анализ процессов…[/{ACCENT}]", spinner="dots"):
                report = collect_processes()
            _render_processes(console, report)
            _record_diagnostic(
                runtime,
                "processes",
                f"Всего процессов: {report.total}, зомби: {len(report.zombies)}",
                asdict(report),
            )
        if storage:
            did_diagnostic = True
            with console.status(f"[{ACCENT}]Анализ дисков…[/{ACCENT}]", spinner="dots"):
                report = collect_storage(include_usage=True)
            _render_storage(console, report)
            _record_diagnostic(
                runtime,
                "storage",
                f"ФС: {len(report.disks)}, Btrfs: {report.has_btrfs}",
                asdict(report),
            )
        if docker:
            did_diagnostic = True
            with console.status(f"[{ACCENT}]Проверка Docker…[/{ACCENT}]", spinner="dots"):
                report = collect_services()
            _render_services(console, report, docker_only=True)
            _record_diagnostic(
                runtime,
                "docker",
                f"Docker: {report.docker.service_state}",
                asdict(report.docker),
            )
        if services:
            did_diagnostic = True
            with console.status(f"[{ACCENT}]Проверка systemd…[/{ACCENT}]", spinner="dots"):
                report = collect_services()
            _render_services(console, report)
            _record_diagnostic(
                runtime,
                "services",
                f"systemd: {report.system_state}, failed: {len(report.failed_units)}",
                asdict(report),
            )

        if chat:
            _chat(runtime, provider=provider, model=model)
            return
        if query:
            if did_diagnostic:
                console.print(
                    f"[{WARN}]Диагностика выполнена. Для AI-запроса запустите без флагов.[/{WARN}]"
                )
                return
            _execute_query(
                runtime,
                " ".join(query),
                provider=provider,
                model=model,
                allow_exec=not no_exec,
            )
            return
        if not did_diagnostic:
            _dashboard(runtime)
    except KeyboardInterrupt:
        console.print("\n" + str(Text("Прервано.", style=MUTED)))
        raise typer.Exit(130) from None


@command_app.command("setup")
def setup_command() -> None:
    """Настроить бесплатные AI-провайдеры."""
    runtime = _load_runtime()
    _setup(runtime)


@command_app.command("models")
def models_command(
    provider: str | None = typer.Option(None, "--provider", help="Только один провайдер."),
) -> None:
    """Показать доступные модели провайдеров."""
    runtime = _load_runtime()
    _show_models(runtime, provider=provider)


@command_app.command("history")
def history_command(
    limit: int = typer.Option(20, "--limit", min=1, max=200, help="Сколько диалогов показать."),
    show: int | None = typer.Option(None, "--show", help="Показать диалог по ID."),
    clear: bool = typer.Option(False, "--clear", help="Очистить историю."),
    yes: bool = typer.Option(False, "--yes", help="Не спрашивать подтверждение при очистке."),
) -> None:
    """История разговоров, команд и диагностики."""
    runtime = _load_runtime()
    if clear:
        _clear_history(runtime, assume_yes=yes)
    else:
        _show_history(runtime, limit=limit, show_id=show)


@command_app.command("config")
def config_command(
    init: bool = typer.Option(False, "--init", help="Создать конфигурацию по умолчанию."),
) -> None:
    """Показать или создать конфигурацию."""
    if init:
        _init_config()
        return
    runtime = _load_runtime()
    _show_config(runtime)


@command_app.command("doctor")
def doctor_command() -> None:
    """Проверить окружение и интеграцию."""
    runtime = _load_runtime()
    _doctor(runtime)


def main() -> None:
    argv = sys.argv[1:]
    try:
        if argv and argv[0] in _SUBCOMMANDS:
            prog_name = Path(sys.argv[0]).name or "ryoku-ai"
            command_app(
                args=argv,
                prog_name=prog_name,
                standalone_mode=True,
            )
            return
        app(prog_name="ryoku-ai")
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
