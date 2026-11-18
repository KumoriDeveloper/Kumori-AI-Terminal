"""Системный промпт и разбор структурированного JSON-ответа."""

from __future__ import annotations

from ryoku_ai.config import Config
from ryoku_ai.execution.models import AIAction, ProposedCommand
from ryoku_ai.execution.planner import (
    build_analysis_prompt,
    build_explain_prompt,
    build_system_prompt,
    looks_like_json,
    parse_action,
)


def test_system_prompt_contains_safety_rules() -> None:
    prompt = build_system_prompt(Config(max_commands=2), "контекст-данные")
    assert "НЕ выполняешь команды" in prompt
    assert "sudo" in prompt
    assert "sh -c" in prompt
    assert "не более 2 команд" in prompt
    assert "контекст-данные" in prompt
    assert "СТРОГО один JSON" in prompt
    assert '"actions": []' in prompt
    assert 'paccache ["-r","-k","3"]' in prompt
    assert "pacman -Scc" in prompt
    assert 'ryoku-temp-inspector ["~/.cache"]' in prompt


def test_explain_and_analysis_prompts() -> None:
    explain = build_explain_prompt("rm -rf /tmp/x")
    assert "rm -rf /tmp/x" in explain
    assert "Не выполняй" in explain
    assert "пустые массивы" in explain
    assert "actions" in explain
    analysis = build_analysis_prompt("$ ps aux\n...")
    assert "$ ps aux" in analysis
    assert "пустые массивы" in analysis
    assert "actions" in analysis


def test_parse_plain_text() -> None:
    action = parse_action("Просто ответ без JSON")
    assert action.message == "Просто ответ без JSON"
    assert action.commands == []


def test_parse_empty_text() -> None:
    action = parse_action("   ")
    assert "пустой ответ" in action.message.lower()


def test_parse_fenced_json() -> None:
    action = parse_action(
        """```json
        {"message": "Проверим сеть.", "commands": [
          {"argv": ["ss", "-tunap"], "description": "Сокеты", "risk": "low"}
        ]}
        ```"""
    )
    assert action.message == "Проверим сеть."
    assert len(action.commands) == 1
    assert action.commands[0].argv == ["ss", "-tunap"]
    assert action.commands[0].risk == "low"


def test_json_embedded_in_plain_text_is_not_executable() -> None:
    action = parse_action(
        'Вот план: {"message": "ok", "commands": [{"argv": ["ps", "aux"]}]} конец'
    )
    assert action.message.startswith("Вот план:")
    assert action.commands == []
    assert action.actions == []


def test_parse_string_command_is_shlex_split() -> None:
    action = parse_action('{"commands": [{"argv": "ps aux"}]}')
    assert action.commands[0].argv == ["ps", "aux"]


def test_parse_command_key_alias() -> None:
    action = parse_action('{"commands": [{"command": "ip -j addr show"}]}')
    assert action.commands[0].argv == ["ip", "-j", "addr", "show"]


def test_parse_invalid_items_are_rejected_not_executed() -> None:
    action = parse_action(
        """{"commands": [
            {"argv": []},
            {"argv": [1, 2]},
            {"command": "ps 'unterminated"}
        ]}"""
    )
    assert action.commands == []
    assert len(action.rejected) == 3


def test_parse_invalid_risk_is_normalized_to_low() -> None:
    action = parse_action('{"commands": [{"argv": ["ps"], "risk": "critical"}]}')
    assert action.commands[0].risk == "low"


def test_parse_commands_default_message() -> None:
    action = parse_action('{"commands": [{"argv": ["ps", "aux"]}]}')
    assert action.message == "Предлагаю выполнить проверку."


def test_parse_non_list_commands_is_empty() -> None:
    action = parse_action('{"message": "ok", "commands": "ps"}')
    assert action.message == "ok"
    assert action.commands == []


def test_parse_structured_action_and_reject_invalid_action() -> None:
    action = parse_action(
        '{"message":"Анализ кеша","actions":[ '
        '{"title":"Анализ пользовательского кеша",'
        '"description":"Только размер","executable":"du",'
        '"arguments":["-sh","--","~/.cache"],'
        '"required_privileges":[],"risk_level":"low",'
        '"expected_effect":"Покажет размер","rollback_available":false},'
        '{"title":"bad","description":"bad","executable":"rm",'
        '"arguments":["-rf","/"],"required_privileges":[],'
        '"risk_level":"critical","expected_effect":"bad",'
        '"rollback_available":true}]}'
    )
    assert len(action.actions) == 1
    assert action.actions[0].executable == "du"
    assert action.commands == []
    assert action.rejected == ["Действие не прошло проверку структуры"]


def test_plain_text_never_becomes_action() -> None:
    action = parse_action("Удалить временные файлы командой rm -rf /tmp")
    assert action.actions == []
    assert action.commands == []


def test_parse_truncates_long_message() -> None:
    action = parse_action("x" * 30000)
    assert len(action.message) == 20000


def test_looks_like_json() -> None:
    assert looks_like_json('{"a": 1}')
    assert looks_like_json("  [1]")
    assert not looks_like_json("text")


def test_action_limit() -> None:
    action = AIAction(
        message="",
        commands=[
            ProposedCommand(argv=["ps"]),
            ProposedCommand(argv=["ip", "addr"]),
            ProposedCommand(argv=["df"]),
        ],
    )
    limited = action.limited(2)
    assert len(limited.commands) == 2
    assert limited.rejected
    assert action.limited(0).commands == []
    assert action.limited(5) is action
