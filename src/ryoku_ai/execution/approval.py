"""Интерактивное подтверждение каждой команды."""

from __future__ import annotations

import shlex
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .cleanup import CleanupActionValidator, ValidatedCleanupAction
from .models import ProposedCommand, ValidatedCommand, ValidationError
from .validator import CommandValidator

Choice = Literal["execute", "edit", "cancel"]


@dataclass(slots=True)
class ApprovalOutcome:
    command: ValidatedCommand | None
    approved: bool
    reason: str = ""


@dataclass(slots=True)
class ActionApprovalOutcome:
    action: ValidatedCleanupAction | None
    approved: bool
    stop_plan: bool = False
    reason: str = ""


class ApprovalEngine:
    def __init__(
        self,
        console: Console,
        validator: CommandValidator,
        *,
        allow_sudo: bool = True,
        input_fn: Callable[[str], str] | None = None,
        interactive: bool | None = None,
    ):
        self.console = console
        self.validator = validator
        self.allow_sudo = allow_sudo
        self._input_fn = input_fn
        if interactive is None:
            interactive = input_fn is not None or sys.stdin.isatty()
        self.interactive = interactive

    def _ask(self, prompt: str) -> str:
        if self._input_fn is not None:
            return self._input_fn(prompt)
        return input(prompt)

    def _choice(self) -> Choice:
        while True:
            try:
                answer = self._ask("Выбор [1/2/3]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return "cancel"
            if answer in {"1", "y", "yes", "д", "да", "выполнить"}:
                return "execute"
            if answer in {"2", "e", "edit", "изменить"}:
                return "edit"
            if answer in {"3", "n", "no", "н", "нет", "отмена", "cancel"}:
                return "cancel"
            self.console.print("[yellow]Введите 1, 2 или 3.[/yellow]")

    def _render_command(self, command: ValidatedCommand, index: int, total: int) -> None:
        risk_colors = {"low": "green", "medium": "yellow", "high": "red"}
        color = risk_colors.get(command.risk, "yellow")
        header = Text("RYOKU AI — COMMAND APPROVAL", style="bold white on dark_red")
        body = Text()
        body.append("Предлагаемое действие:\n", style="bold")
        body.append(command.description + "\n\n")
        body.append("Команда:\n", style="bold")
        body.append(command.display(), style="bold bright_cyan")
        body.append("\n\nПоследствия:\n", style="bold")
        body.append(command.policy_note + "\n")
        if command.requires_sudo:
            body.append(
                "\n⚠ Требуются права root. Будет отдельное подтверждение sudo.\n",
                style="bold yellow",
            )
        body.append("\n")
        body.append("[1] Выполнить    [2] Изменить    [3] Отмена", style="bold green")
        if total > 1:
            body.append(f"\nКоманда {index} из {total}", style="dim")
        self.console.print()
        self.console.print(
            Panel(
                body,
                title=header,
                subtitle=Text(f"риск: {command.risk}", style=color),
                border_style=color,
                padding=(1, 2),
            )
        )

    def _confirm_sudo(self, command: ValidatedCommand) -> bool:
        panel = Panel(
            Text.from_markup(
                "[bold yellow]Требуется повышение прав[/bold yellow]\n\n"
                "Команда будет запущена как:\n"
                f"[bold bright_cyan]{command.display()}[/bold bright_cyan]\n\n"
                "Используется [bold]sudo -n[/bold] — без запроса пароля. "
                "Если для sudo нужен пароль, выполнение не начнётся. "
                "Пароль никогда не передаётся приложению.\n\n"
                "[1] Да, выполнить с sudo    [2] Нет, отменить"
            ),
            border_style="yellow",
            title=Text("SUDO APPROVAL", style="bold black on yellow"),
            padding=(1, 2),
        )
        self.console.print(panel)
        while True:
            try:
                answer = self._ask("Подтвердить sudo? [1/2]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
            if answer in {"1", "y", "yes", "д", "да"}:
                return True
            if answer in {"2", "n", "no", "н", "нет", ""}:
                return False
            self.console.print("[yellow]Введите 1 или 2.[/yellow]")

    def _edit(self, command: ValidatedCommand) -> ValidatedCommand | None:
        self.console.print(
            "\n[bold]Новая команда[/bold] (argv через пробел, без sudo, "
            "pipe, ; и перенаправлений):"
        )
        while True:
            try:
                line = self._ask("> ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if not line:
                return None
            try:
                parts = shlex.split(line, posix=True)
            except ValueError as exc:
                self.console.print(f"[red]Ошибка разбора: {exc}[/red]")
                continue
            try:
                raw = ProposedCommand(
                    argv=parts,
                    description=command.description,
                    risk=command.risk,
                    requires_sudo=command.requires_sudo,
                )
                return self.validator.validate(raw)
            except ValidationError as exc:
                self.console.print(f"[red]Команда отклонена: {exc.reason}[/red]")
                self.console.print("[dim]Введите другую команду или оставьте пустую строку.[/dim]")

    def request_approval(
        self,
        command: ValidatedCommand,
        *,
        index: int = 1,
        total: int = 1,
    ) -> ApprovalOutcome:
        if not self.interactive:
            self.console.print(
                Panel(
                    Text(
                        "Нет интерактивного терминала. Команда НЕ будет выполнена.\n"
                        "Запустите Ryoku AI Terminal в терминале, чтобы подтвердить действие.",
                        style="yellow",
                    ),
                    border_style="yellow",
                )
            )
            return ApprovalOutcome(command, False, "нет TTY")

        current = command
        for _ in range(5):
            self._render_command(current, index, total)
            choice = self._choice()
            if choice == "cancel":
                self.console.print("[yellow]Действие отменено.[/yellow]")
                return ApprovalOutcome(current, False, "отменено пользователем")
            if choice == "edit":
                edited = self._edit(current)
                if edited is None:
                    self.console.print("[yellow]Изменение отменено.[/yellow]")
                    return ApprovalOutcome(current, False, "изменение отменено")
                current = edited
                continue

            if current.requires_sudo:
                if not self.allow_sudo:
                    self.console.print("[red]sudo отключён в конфигурации.[/red]")
                    return ApprovalOutcome(current, False, "sudo отключён")
                if not self._confirm_sudo(current):
                    self.console.print("[yellow]Sudo-выполнение отменено.[/yellow]")
                    return ApprovalOutcome(current, False, "sudo отменён")

            self.console.print("[green]Подтверждено. Запускаю…[/green]")
            return ApprovalOutcome(current, True, "подтверждено")

        self.console.print("[red]Слишком много попыток изменения. Отменено.[/red]")
        return ApprovalOutcome(current, False, "слишком много попыток")

    def request_action_approval(
        self,
        action: ValidatedCleanupAction,
        action_validator: CleanupActionValidator,
        *,
        index: int,
        total: int,
        task: str | None = None,
    ) -> ActionApprovalOutcome:
        if not self.interactive:
            self.console.print(
                Panel(
                    Text("Нет интерактивного терминала. Действие НЕ будет выполнено."),
                    border_style="yellow",
                )
            )
            return ActionApprovalOutcome(action, False, reason="нет TTY")

        current = action
        for _ in range(5):
            color = {"low": "green", "medium": "yellow", "high": "red"}[current.risk_level]
            body = Text()
            body.append(f"Задача: {task or current.title}\n", style="bold")
            body.append(f"Действие: {current.title}\n\n", style="bold")
            body.append(f"Действие {index} из {total}\n", style="dim")
            body.append(f"{current.description}\n\n")
            body.append("Операция:\n", style="bold")
            body.append(f"{current.display()}\n\n", style="bold bright_cyan")
            body.append("Ожидаемый эффект:\n", style="bold")
            body.append(current.expected_effect + "\n")
            body.append(f"\nРиск: {current.risk_level}\n", style=color)
            if current.requires_sudo:
                body.append("Требуются права root; будет отдельное подтверждение.\n", style="bold yellow")
            body.append("\n[Enter] Подтвердить  [2] Изменить  [3] Пропустить  [4] Отмена", style="bold")
            self.console.print(
                Panel(
                    body,
                    title=Text("RYOKU AI — ACTION APPROVAL", style="bold white on dark_red"),
                    border_style=color,
                    padding=(1, 2),
                )
            )
            try:
                answer = self._ask("Выбор [Enter/2/3/4]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return ActionApprovalOutcome(current, False, True, "план отменён")
            if answer in {"", "1", "y", "yes", "д", "да"}:
                if current.requires_sudo:
                    if not self.allow_sudo:
                        return ActionApprovalOutcome(current, False, reason="sudo отключён")
                    if not self._confirm_action_sudo(current):
                        return ActionApprovalOutcome(current, False, reason="sudo отменён")
                return ActionApprovalOutcome(current, True, reason="подтверждено")
            if answer == "3":
                return ActionApprovalOutcome(current, False, reason="пропущено")
            if answer == "4":
                return ActionApprovalOutcome(current, False, True, "план отменён")
            if answer != "2":
                self.console.print("[yellow]Введите Enter, 2, 3 или 4.[/yellow]")
                continue

            try:
                line = self._ask("Новая операция (разрешённые аргументы будут проверены): ").strip()
            except (EOFError, KeyboardInterrupt):
                return ActionApprovalOutcome(current, False, reason="изменение отменено")
            if not line:
                return ActionApprovalOutcome(current, False, reason="изменение отменено")
            try:
                parts = shlex.split(line, posix=True)
                if not parts:
                    raise ValueError("пустая команда")
                proposed = action_validator.proposed_for(parts[0], parts[1:])
                current = action_validator.validate(proposed)
            except (ValueError, ValidationError) as exc:
                reason = exc.reason if isinstance(exc, ValidationError) else str(exc)
                self.console.print(f"[red]Действие отклонено: {reason}[/red]")
        return ActionApprovalOutcome(current, False, reason="слишком много попыток")

    def _confirm_action_sudo(self, action: ValidatedCleanupAction) -> bool:
        self.console.print(
            Panel(
                Text(
                    f"Действие потребует root через sudo -n:\n\n{action.display()}\n\n"
                    "Пароль не запрашивается и не передаётся программе.\n"
                    "[1] Подтвердить sudo  [2] Отмена"
                ),
                title=Text("SUDO APPROVAL", style="bold black on yellow"),
                border_style="yellow",
            )
        )
        try:
            answer = self._ask("Подтвердить sudo? [1/2]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in {"1", "y", "yes", "д", "да"}
