"""Подтверждение команд: отмена, изменение и отдельное подтверждение sudo."""

from __future__ import annotations

import io

from rich.console import Console

from ryoku_ai.execution.approval import ApprovalEngine
from ryoku_ai.execution.models import ProposedCommand
from ryoku_ai.execution.validator import CommandValidator


def make_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def make_validator() -> CommandValidator:
    return CommandValidator(resolver=lambda name: f"/usr/bin/{name}")


def make_command(*, sudo: bool = False):
    validator = make_validator()
    return validator, validator.validate(
        ProposedCommand(
            argv=["ss", "-tunap"] if sudo else ["ps", "aux"],
            description="Проверка",
            requires_sudo=sudo,
        )
    )


def test_non_interactive_never_executes() -> None:
    validator, command = make_command()
    engine = ApprovalEngine(
        make_console(),
        validator,
        interactive=False,
    )
    outcome = engine.request_approval(command)
    assert outcome.approved is False
    assert outcome.reason == "нет TTY"


def test_simple_approval() -> None:
    validator, command = make_command()
    answers = iter(["1"])
    engine = ApprovalEngine(
        make_console(),
        validator,
        input_fn=lambda _prompt: next(answers),
        interactive=True,
    )
    outcome = engine.request_approval(command)
    assert outcome.approved is True
    assert outcome.command is not None
    assert outcome.command.argv == ("ps", "aux")


def test_cancel() -> None:
    validator, command = make_command()
    answers = iter(["3"])
    engine = ApprovalEngine(
        make_console(),
        validator,
        input_fn=lambda _prompt: next(answers),
        interactive=True,
    )
    outcome = engine.request_approval(command)
    assert outcome.approved is False
    assert "отменено" in outcome.reason


def test_sudo_requires_second_confirmation() -> None:
    validator, command = make_command(sudo=True)
    answers = iter(["1", "2"])
    engine = ApprovalEngine(
        make_console(),
        validator,
        input_fn=lambda _prompt: next(answers),
        interactive=True,
    )
    outcome = engine.request_approval(command)
    assert outcome.approved is False
    assert outcome.reason == "sudo отменён"


def test_sudo_approved_after_both_confirmations() -> None:
    validator, command = make_command(sudo=True)
    answers = iter(["1", "1"])
    engine = ApprovalEngine(
        make_console(),
        validator,
        input_fn=lambda _prompt: next(answers),
        interactive=True,
    )
    outcome = engine.request_approval(command)
    assert outcome.approved is True
    assert outcome.command is not None
    assert outcome.command.requires_sudo is True


def test_sudo_disabled_in_config_cancels() -> None:
    validator, command = make_command(sudo=True)
    answers = iter(["1"])
    engine = ApprovalEngine(
        make_console(),
        validator,
        input_fn=lambda _prompt: next(answers),
        interactive=True,
        allow_sudo=False,
    )
    outcome = engine.request_approval(command)
    assert outcome.approved is False
    assert outcome.reason == "sudo отключён"


def test_edit_command_is_revalidated() -> None:
    validator, command = make_command()
    answers = iter(["2", "ps -e", "1"])
    engine = ApprovalEngine(
        make_console(),
        validator,
        input_fn=lambda _prompt: next(answers),
        interactive=True,
    )
    outcome = engine.request_approval(command)
    assert outcome.approved is True
    assert outcome.command is not None
    assert outcome.command.argv == ("ps", "-e")


def test_invalid_edit_can_be_cancelled() -> None:
    validator, command = make_command()
    answers = iter(["2", "rm -rf /", ""])
    engine = ApprovalEngine(
        make_console(),
        validator,
        input_fn=lambda _prompt: next(answers),
        interactive=True,
    )
    outcome = engine.request_approval(command)
    assert outcome.approved is False
    assert outcome.reason == "изменение отменено"


def test_unknown_choice_reasks() -> None:
    validator, command = make_command()
    answers = iter(["maybe", "1"])
    engine = ApprovalEngine(
        make_console(),
        validator,
        input_fn=lambda _prompt: next(answers),
        interactive=True,
    )
    outcome = engine.request_approval(command)
    assert outcome.approved is True


def test_eof_is_treated_as_cancel() -> None:
    validator, command = make_command()

    def raise_eof(_prompt: str) -> str:
        raise EOFError

    engine = ApprovalEngine(
        make_console(),
        validator,
        input_fn=raise_eof,
        interactive=True,
    )
    outcome = engine.request_approval(command)
    assert outcome.approved is False


def test_interactive_defaults_to_input_fn_presence() -> None:
    validator, _command = make_command()
    engine = ApprovalEngine(
        make_console(),
        validator,
        input_fn=lambda _prompt: "3",
    )
    assert engine.interactive is True
