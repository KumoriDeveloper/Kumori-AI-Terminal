"""Fish-интеграция: автозагрузка, аргументы и отсутствие рекурсии."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

FISH = shutil.which("fish")
FUNCTION = Path(__file__).resolve().parents[1] / "fish" / "functions" / "ai.fish"

pytestmark = pytest.mark.skipif(FISH is None, reason="fish не установлен")


def test_fish_function_file_exists_and_has_marker() -> None:
    text = FUNCTION.read_text(encoding="utf-8")
    assert "Ryoku AI Terminal" in text
    assert "command ryoku-ai $argv" in text


def test_fish_syntax_is_valid() -> None:
    result = subprocess.run(
        [FISH, "-n", str(FUNCTION)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_fish_wrapper_passes_cyrillic_arguments(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "ryoku-ai"
    fake.write_text(
        "#!/bin/sh\nfor arg in \"$@\"; do printf '<%s>\\n' \"$arg\"; done\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    command = f"source {FUNCTION}; ai 'привет мир' --provider groq"
    result = subprocess.run(
        [FISH, "--no-config", "-c", command],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["<привет мир>", "<--provider>", "<groq>"]


def test_fish_wrapper_without_arguments(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "ryoku-ai"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    result = subprocess.run(
        [FISH, "--no-config", "-c", f"source {FUNCTION}; ai"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_fish_wrapper_reports_missing_binary(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PATH"] = str(tmp_path)
    result = subprocess.run(
        [FISH, "--no-config", "-c", f"source {FUNCTION}; ai test"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 127
    assert "ryoku-ai не найдена" in result.stderr
