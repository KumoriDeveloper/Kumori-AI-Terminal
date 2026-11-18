"""Read-only temporary directory inspection tests use temporary paths and mocks."""

from __future__ import annotations

import os
from pathlib import Path

from ryoku_ai.system.temp_inspector import inspect_directory, inspect_standard_temp_directories


def test_empty_directory_has_successful_zero_content_size(tmp_path: Path) -> None:
    directory = tmp_path / "empty"
    directory.mkdir()

    report = inspect_directory(directory)

    assert report.status == "success"
    assert report.inaccessible_directories == 0
    assert report.readable_bytes == directory.stat().st_blocks * 512


def test_missing_directory_is_reported_without_scanning(tmp_path: Path) -> None:
    report = inspect_directory(tmp_path / "missing")

    assert report.status == "failed"
    assert "не найден" in report.summary
    assert report.scanned_entries == 0


def test_permission_denied_directories_are_counted(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "denied-a").mkdir(parents=True)
    (root / "denied-b").mkdir()
    readable = root / "readable"
    readable.mkdir()
    (readable / "file").write_bytes(b"data")
    real_scandir = os.scandir

    def mocked_scandir(path):
        resolved = Path(os.readlink(f"/proc/self/fd/{path}")) if isinstance(path, int) else Path(path)
        if resolved.name.startswith("denied-"):
            raise PermissionError("mocked permission denied")
        return real_scandir(path)

    monkeypatch.setattr("ryoku_ai.system.temp_inspector.os.scandir", mocked_scandir)

    report = inspect_directory(root)

    assert report.status == "partial_success"
    assert report.inaccessible_directories == 2
    assert any("Недоступные каталоги: 2" in warning for warning in report.warnings)
    assert report.readable_bytes >= (readable / "file").stat().st_blocks * 512


def test_files_disappearing_during_scan_are_partial(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "vanishing").write_text("data", encoding="utf-8")

    real_stat = os.stat

    def mocked_stat(path, *args, **kwargs):
        if path == "vanishing" and kwargs.get("dir_fd") is not None:
            raise FileNotFoundError("mocked unlink race")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr("ryoku_ai.system.temp_inspector.os.stat", mocked_stat)

    report = inspect_directory(root)

    assert report.status == "partial_success"
    assert report.disappeared_entries == 1
    assert any("исчезли во время сканирования: 1" in warning for warning in report.warnings)


def test_symlinks_are_counted_but_not_followed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "large-file").write_bytes(b"x" * 1024 * 1024)
    (root / "link-outside").symlink_to(outside, target_is_directory=True)
    expected = root.lstat().st_blocks * 512 + (root / "link-outside").lstat().st_blocks * 512

    report = inspect_directory(root)

    assert report.status == "success"
    assert report.readable_bytes == expected


def test_very_long_paths_are_scanned_without_render_assumptions(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    current = root
    for index in range(10):
        current = current / (f"segment-{index}-" + "x" * 180)
        current.mkdir()
    (current / "data").write_bytes(b"long path data")

    report = inspect_directory(current, max_depth=16)

    assert report.status == "success"
    assert len(report.path) > 1600
    assert report.readable_bytes > 0


def test_standard_targets_use_requested_home(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".cache").mkdir(parents=True)

    reports = inspect_standard_temp_directories(home=home)

    assert [report.path for report in reports] == ["/tmp", "/var/tmp", str(home / ".cache")]
    assert reports[2].status == "success"


def test_symlink_root_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    report = inspect_directory(link)

    assert report.status == "failed"
    assert "symlink" in report.summary


def test_directory_replaced_by_symlink_during_scan_is_not_followed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    child = root / "child"
    outside = tmp_path / "outside"
    child.mkdir(parents=True)
    outside.mkdir()
    large_file = outside / "must-not-count"
    large_file.write_bytes(b"x" * (2 * 1024 * 1024))
    real_open = os.open
    replaced = False

    def replace_before_open(path, flags, *args, dir_fd=None, **kwargs):
        nonlocal replaced
        if path == "child" and dir_fd is not None and not replaced:
            replaced = True
            child.rmdir()
            child.symlink_to(outside, target_is_directory=True)
        if dir_fd is None:
            return real_open(path, flags, *args, **kwargs)
        return real_open(path, flags, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr("ryoku_ai.system.temp_inspector.os.open", replace_before_open)

    report = inspect_directory(root)

    assert replaced is True
    assert report.status == "partial_success"
    assert report.disappeared_entries == 1
    assert report.readable_bytes < large_file.stat().st_blocks * 512


def test_scan_stops_at_configured_entry_limit(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    for index in range(10):
        (root / f"item-{index}").write_text("x", encoding="utf-8")

    report = inspect_directory(root, max_entries=3)

    assert report.status == "partial_success"
    assert report.scanned_entries == 3
    assert any("лимит обхода" in warning for warning in report.warnings)
