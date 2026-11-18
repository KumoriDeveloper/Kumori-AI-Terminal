"""Строгая проверка команд: белый список, argv, пути и sudo."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ryoku_ai.execution.models import ProposedCommand, ValidationError
from ryoku_ai.execution.validator import CommandValidator


def validator(**kwargs) -> CommandValidator:
    options = {"resolver": lambda name: f"/usr/bin/{name}"}
    options.update(kwargs)
    return CommandValidator(**options)


def check(argv: list[str], **kwargs):
    return validator(**kwargs).validate(ProposedCommand(argv=argv))


@pytest.mark.parametrize(
    "argv",
    [
        ["ps", "aux"],
        ["ss", "-tunap"],
        ["ip", "-j", "addr", "show"],
        ["journalctl", "-b", "-p", "3", "-n", "50", "--no-pager"],
        ["docker", "ps", "-a"],
        ["docker", "images"],
        ["pacman", "-Qi", "bash"],
        ["cat", "/etc/os-release"],
        ["ls", "/etc"],
        ["df", "-h"],
        ["whoami"],
        ["uptime"],
        ["uname", "-a"],
        ["free", "-h"],
        ["lsblk", "-f"],
        ["findmnt", "--json"],
        ["sensors"],
        ["ip", "route", "show", "default"],
        ["ping", "1.1.1.1"],
        ["systemctl", "status", "docker"],
        ["journalctl", "-u", "docker.service", "-n", "20"],
        ["btrfs", "filesystem", "df", "/"],
        ["du", "-s", "/etc"],
    ],
)
def test_read_only_commands_are_allowed(argv: list[str]) -> None:
    command = check(argv)
    assert command.argv[0] == argv[0]
    assert command.executable.startswith("/usr/bin/")
    assert command.digest
    assert command.risk in {"low", "medium", "high"}


def test_ping_gets_safe_default_count() -> None:
    command = check(["ping", "1.1.1.1"])
    assert command.argv[:3] == ("ping", "-c", "4")


def test_docker_stats_gets_no_stream() -> None:
    command = check(["docker", "stats", "my-container"])
    assert "--no-stream" in command.argv


def test_safe_file_reads_only_allowed_files(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Чтение этого файла"):
        check(["cat", "/etc/passwd"])
    with pytest.raises(ValidationError, match="секретн"):
        check(["cat", "/etc/shadow"])
    with pytest.raises(ValidationError, match="Чтение этого файла"):
        check(["cat", str(tmp_path / "private.txt")])


@pytest.mark.parametrize(
    "argv",
    [
        ["sh", "-c", "id"],
        ["bash", "-c", "id"],
        ["rm", "-rf", "/"],
        ["sudo", "id"],
        ["python3", "-c", "print(1)"],
        ["perl", "-e", "1"],
        ["sed", "-n", "1p", "/etc/passwd"],
        ["git", "status"],
        ["npm", "install"],
        ["curl", "https://example.com"],
        ["wget", "https://example.com"],
        ["nc", "-l", "1234"],
        ["ssh", "host"],
        ["kill", "1"],
        ["mount", "/dev/sda1", "/mnt"],
        ["systemctl", "start", "docker"],
        ["systemctl", "restart", "docker"],
        ["systemctl", "enable", "docker"],
        ["pacman", "-S", "nano"],
        ["pacman", "-R", "nano"],
        ["ip", "link", "set", "lo", "up"],
        ["ip", "addr", "add", "1.2.3.4/32", "dev", "lo"],
        ["docker", "run", "alpine"],
        ["docker", "rm", "container"],
        ["btrfs", "subvolume", "delete", "/"],
        ["nvidia-smi", "--gpu-reset"],
    ],
)
def test_dangerous_or_mutating_commands_are_blocked(argv: list[str]) -> None:
    with pytest.raises(ValidationError):
        check(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["ps", "aux;", "id"],
        ["ps", "aux|id"],
        ["ps", "aux&id"],
        ["ps", "`id`"],
        ["ps", "$(id)"],
        ["ps", "a>b"],
        ["ps", "a<b"],
        ["ls", "*"],
        ["ls", "?"],
        ["ls", "[abc]"],
        ["ls", "{a,b}"],
        ["ls", "line\nnext"],
        ["ls", "tab\tchar"],
    ],
)
def test_shell_metacharacters_and_globs_are_blocked(argv: list[str]) -> None:
    with pytest.raises(ValidationError):
        check(argv)


def test_absolute_path_and_unknown_binary_are_blocked() -> None:
    with pytest.raises(ValidationError, match="простым именем"):
        check(["/usr/bin/ps", "aux"])
    with pytest.raises(ValidationError, match="белый список"):
        check(["find", "/tmp", "-name", "x"])
    with pytest.raises(ValidationError, match="белый список"):
        check(["echo", "hello"])


def test_empty_and_oversized_argv_are_blocked() -> None:
    with pytest.raises(ValidationError, match="Пустая команда"):
        validator().validate(ProposedCommand.model_construct(argv=[]))
    with pytest.raises(ValidationError, match="Слишком длинный аргумент"):
        check(["ps", "a" * 1001])
    with pytest.raises(ValidationError, match="Слишком много аргументов"):
        check(["ps", *["-e"] * 32])
    with pytest.raises(ValidationError, match="строками"):
        validator().validate(ProposedCommand.model_construct(argv=["ps", 1]))


def test_invalid_flags_are_blocked() -> None:
    with pytest.raises(ValidationError, match="Флаг --evil"):
        check(["ps", "--evil"])
    with pytest.raises(ValidationError, match="требует значение"):
        check(["journalctl", "-u"])
    with pytest.raises(ValidationError, match="диапазон"):
        check(["ping", "-c", "99", "1.1.1.1"])


def test_path_traversal_and_sensitive_paths_are_blocked() -> None:
    with pytest.raises(ValidationError, match="секретн"):
        check(["cat", "/etc/../etc/shadow"])
    with pytest.raises(ValidationError, match="секретн"):
        check(["ls", "~/.ssh"])
    with pytest.raises(ValidationError, match="вне разрешённых"):
        check(["ls", "/root"])


def test_missing_executable_is_blocked() -> None:
    instance = validator(resolver=lambda name: None)
    with pytest.raises(ValidationError, match="не найдена"):
        instance.validate(ProposedCommand(argv=["ps", "aux"]))


def test_sudo_requires_separate_approval_flag() -> None:
    command = check(["ss", "-tunap"], allow_sudo=True)
    assert command.requires_sudo is False
    sudo_command = validator(allow_sudo=True).validate(
        ProposedCommand(argv=["ss", "-tunap"], requires_sudo=True)
    )
    assert sudo_command.requires_sudo is True
    assert sudo_command.display().startswith("sudo -n -- ")

    with pytest.raises(ValidationError, match="Повышение прав отключено"):
        validator(allow_sudo=False).validate(
            ProposedCommand(argv=["ss", "-tunap"], requires_sudo=True)
        )


def test_sudo_is_ignored_for_tools_that_do_not_need_it() -> None:
    command = validator(allow_sudo=True).validate(
        ProposedCommand(argv=["ps", "aux"], requires_sudo=True)
    )
    assert command.requires_sudo is False
    assert "не требуется" in command.policy_note


def test_revalidate_detects_modified_command() -> None:
    instance = validator()
    command = instance.validate(ProposedCommand(argv=["ps", "aux"]))
    modified = replace(command, argv=("ps", "-e"))
    with pytest.raises(ValidationError) as exc_info:
        instance.revalidate(modified)
    assert exc_info.value.category == "integrity"


def test_revalidate_accepts_unchanged_command() -> None:
    instance = validator()
    command = instance.validate(ProposedCommand(argv=["ps", "aux"]))
    fresh = instance.revalidate(command)
    assert fresh.digest == command.digest


def test_validate_line_parses_and_rechecks() -> None:
    instance = validator()
    command = instance.validate_line("ps aux")
    assert command.argv == ("ps", "aux")
    with pytest.raises(ValidationError):
        instance.validate_line("ps aux; rm -rf /")


def test_description_is_truncated_and_falls_back_to_policy_note() -> None:
    instance = validator()
    command = instance.validate(
        ProposedCommand(argv=["ps", "aux"], description="x" * 600)
    )
    assert len(command.description) == 500
    fallback = instance.validate(ProposedCommand(argv=["ps", "aux"]))
    assert "Только чтение" in fallback.policy_note
