"""CLI routes temp-directory queries and last-result explanation safely."""

from __future__ import annotations

import io
import json
from pathlib import Path

from rich.console import Console

import ryoku_ai.cli as cli
from ryoku_ai.ai.base import ChatResponse
from ryoku_ai.config import Config, Credentials
from ryoku_ai.execution.executor import ExecutionResult
from ryoku_ai.execution.models import ProposedCommand
from ryoku_ai.execution.validator import CommandValidator
from ryoku_ai.storage.history import HistoryService
from ryoku_ai.system._run import format_bytes
from ryoku_ai.system.temp_inspector import DirectoryInspection


def test_direct_temp_query_uses_builtin_inspector_without_ai(tmp_path: Path, monkeypatch) -> None:
    output = io.StringIO()
    runtime = cli.Runtime(
        Config(),
        Credentials(),
        Console(file=output, force_terminal=False, width=80),
        HistoryService(tmp_path / "history.db", secrets=[]),
        [],
    )
    target = tmp_path / "simulated-tmp"
    target.mkdir()
    monkeypatch.setattr(cli, "_inspection_query_paths", lambda _query: [target])
    monkeypatch.setattr(
        cli,
        "inspect_directory",
        lambda path: DirectoryInspection(
            path=str(path),
            readable_bytes=39_000_000,
            inaccessible_directories=10,
            status="partial_success",
            warnings=["Недоступные каталоги: 10"],
            summary="Измерение неполное.",
        ),
    )
    monkeypatch.setattr(cli, "_build_router", lambda _runtime: (_ for _ in ()).throw(AssertionError("no AI configured")))

    handled = cli._handle_temp_inspection(
        runtime,
        "проанализируй размер /tmp",
        provider=None,
        model=None,
    )

    assert handled is True
    rendered = output.getvalue()
    assert "PARTIAL RESULT" in rendered
    assert format_bytes(39_000_000) in rendered
    assert "Inaccessible directories: 10" in rendered


def test_explain_last_result_sends_saved_partial_result_to_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = io.StringIO()
    history = HistoryService(tmp_path / "history.db", secrets=[])
    conversation_id = history.start_conversation(title="temp scan")
    assert conversation_id is not None
    history.add_command(
        conversation_id,
        argv=["du", "-sh", "/tmp"],
        description="measure temp",
        risk="low",
        requires_sudo=False,
        approved=True,
        executed=True,
        exit_code=1,
        output_preview="39M\t/tmp\n[stderr]\ndu: cannot read directory '/tmp/private': Permission denied",
        note="partial output retained",
    )
    runtime = cli.Runtime(
        Config(),
        Credentials(groq="test-key"),
        Console(file=output, force_terminal=False, width=100),
        history,
        [],
    )
    captured = {}

    class StubRouter:
        credentials = runtime.credentials

    def fake_ask(_console, _router, request, **_kwargs):
        captured["request"] = request
        return ChatResponse(
            text='{"message":"39M измерено частично.","commands":[],"actions":[]}',
            provider="groq",
            model="test-model",
        )

    monkeypatch.setattr(cli, "_build_router", lambda _runtime: StubRouter())
    monkeypatch.setattr(cli, "_ask_ai", fake_ask)

    handled = cli._explain_latest_command(
        runtime,
        "объясни результат последней команды",
        provider=None,
        model=None,
    )

    assert handled is True
    prompt = captured["request"].messages[-1].content
    assert '"status": "partial_success"' in prompt
    assert "39M" in prompt
    assert "Permission denied" in prompt
    assert "39M измерено частично" in output.getvalue()


def test_explain_last_result_preserves_scanner_structured_payload(tmp_path: Path, monkeypatch) -> None:
    history = HistoryService(tmp_path / "history.db", secrets=[])
    conversation_id = history.start_conversation(title="temp scan")
    assert conversation_id is not None
    payload = {
        "path": "/var/tmp",
        "readable_size": "0 B",
        "inaccessible_directories": 10,
        "status": "partial_success",
        "warnings": ["Недоступные каталоги: 10"],
        "summary": "Размер неполный.",
    }
    history.add_command(
        conversation_id,
        argv=["temp-inspect", "/var/tmp"],
        description="scan",
        risk="low",
        requires_sudo=False,
        approved=True,
        executed=True,
        exit_code=1,
        output_preview=json.dumps(payload, ensure_ascii=False),
    )
    runtime = cli.Runtime(
        Config(),
        Credentials(groq="test-key"),
        Console(file=io.StringIO(), force_terminal=False),
        history,
        [],
    )
    captured = {}

    class StubRouter:
        credentials = runtime.credentials

    def fake_ask(_console, _router, request, **_kwargs):
        captured["request"] = request
        return ChatResponse(
            text='{"message":"Ноль доступных байт измерено.","commands":[],"actions":[]}',
            provider="groq",
            model="test-model",
        )

    monkeypatch.setattr(cli, "_build_router", lambda _runtime: StubRouter())
    monkeypatch.setattr(cli, "_ask_ai", fake_ask)
    cli._explain_latest_command(runtime, "объясни результат последней команды", provider=None, model=None)

    prompt = captured["request"].messages[-1].content
    assert '"status": "partial_success"' in prompt
    assert '"inaccessible_directories": 10' in prompt
    assert '"readable_size": "0 B"' in prompt


def test_explain_last_result_without_api_renders_partial_summary(tmp_path: Path, monkeypatch) -> None:
    history = HistoryService(tmp_path / "history.db", secrets=[])
    conversation_id = history.start_conversation(title="temp scan")
    assert conversation_id is not None
    payload = {
        "path": "/var/tmp",
        "readable_bytes": 0,
        "scanned_entries": 10,
        "inaccessible_directories": 10,
        "inaccessible_entries": 0,
        "disappeared_entries": 0,
        "status": "partial_success",
        "warnings": ["Недоступные каталоги: 10"],
        "summary": "Доступный размер: 0 B. Нижняя оценка.",
    }
    history.add_command(
        conversation_id,
        argv=["temp-inspect", "/var/tmp"],
        description="scan",
        risk="low",
        requires_sudo=False,
        approved=True,
        executed=True,
        exit_code=1,
        output_preview=json.dumps(payload, ensure_ascii=False),
    )
    output = io.StringIO()
    runtime = cli.Runtime(
        Config(),
        Credentials(),
        Console(file=output, force_terminal=False, width=100),
        history,
        [],
    )
    monkeypatch.setattr(cli, "_build_router", lambda _runtime: (_ for _ in ()).throw(AssertionError("no AI configured")))

    cli._explain_latest_command(
        runtime,
        "объясни результат последней команды",
        provider=None,
        model=None,
    )

    rendered = output.getvalue()
    assert "PARTIAL RESULT" in rendered
    assert "Readable size: 0 B" in rendered
    assert "Inaccessible directories: 10" in rendered
    assert "zero does not mean the tree is empty" in rendered


def test_post_execution_ai_receives_structured_partial_result(tmp_path: Path, monkeypatch) -> None:
    runtime = cli.Runtime(
        Config(),
        Credentials(groq="test-key"),
        Console(file=io.StringIO(), force_terminal=False),
        HistoryService(tmp_path / "history.db", secrets=[]),
        [],
    )
    session = cli.ConversationSession(runtime.config, runtime.history, title="scan")
    command = CommandValidator(resolver=lambda name: f"/usr/bin/{name}").validate(
        ProposedCommand(argv=["du", "-sh", "/tmp"], description="measure tmp")
    )
    result = ExecutionResult(
        argv=command.argv,
        exit_code=1,
        stdout="39M\t/tmp",
        stderr="du: cannot read directory '/tmp/private': Permission denied",
        duration_ms=11,
    )
    captured = {}

    def fake_ask(_console, _router, request, **_kwargs):
        captured["request"] = request
        return ChatResponse(
            text='{"message":"39M measured approximately.","commands":[],"actions":[]}',
            provider="groq",
            model="test-model",
        )

    monkeypatch.setattr(cli, "_ask_ai", fake_ask)

    cli._analyze_results(
        runtime,
        router=object(),
        session=session,
        executed=[(command, result, result.combined)],
        provider=None,
        model=None,
    )

    prompt = captured["request"].messages[-1].content
    assert '"status": "partial_success"' in prompt
    assert "Недоступных каталогов: 1" in prompt
    assert "приблизительной нижней оценкой" in prompt
