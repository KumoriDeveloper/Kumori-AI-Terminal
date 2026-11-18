"""Диски, Btrfs, inode и крупные каталоги."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ._run import format_bytes, read_text, run_tool

_REAL_FS = {
    "ext2",
    "ext3",
    "ext4",
    "btrfs",
    "xfs",
    "vfat",
    "exfat",
    "ntfs",
    "ntfs3",
    "f2fs",
    "zfs",
    "bcachefs",
}
_PSEUDO_FS = {
    "proc",
    "sysfs",
    "devtmpfs",
    "devpts",
    "tmpfs",
    "ramfs",
    "cgroup",
    "cgroup2",
    "overlay",
    "squashfs",
    "securityfs",
    "debugfs",
    "tracefs",
    "pstore",
    "bpf",
    "autofs",
    "mqueue",
    "hugetlbfs",
    "configfs",
    "fusectl",
    "efivarfs",
    "nsfs",
    "rpc_pipefs",
    "binfmt_misc",
}


@dataclass(slots=True)
class DiskInfo:
    device: str
    mountpoint: str
    fstype: str
    total: int
    used: int
    free: int
    percent: float

    @property
    def total_text(self) -> str:
        return format_bytes(self.total)

    @property
    def used_text(self) -> str:
        return format_bytes(self.used)

    @property
    def free_text(self) -> str:
        return format_bytes(self.free)


@dataclass(slots=True)
class InodeInfo:
    mountpoint: str
    total: int
    free: int
    used_percent: float


@dataclass(slots=True)
class StorageReport:
    disks: list[DiskInfo] = field(default_factory=list)
    btrfs_usage: str = ""
    btrfs_error: str | None = None
    inodes: list[InodeInfo] = field(default_factory=list)
    top_dirs: list[tuple[str, int]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def has_btrfs(self) -> bool:
        return any(disk.fstype == "btrfs" for disk in self.disks)


def _unescape_mount(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _mounts() -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    for line in read_text("/proc/mounts").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        device, mountpoint, fstype = parts[0], _unescape_mount(parts[1]), parts[2]
        entries.append((device, mountpoint, fstype))
    return entries


def collect_disks() -> tuple[list[DiskInfo], list[str]]:
    disks: list[DiskInfo] = []
    errors: list[str] = []
    seen_mounts: set[str] = set()
    for device, mountpoint, fstype in _mounts():
        if fstype not in _REAL_FS or fstype in _PSEUDO_FS:
            continue
        if mountpoint in seen_mounts:
            continue
        try:
            usage = shutil.disk_usage(mountpoint)
        except OSError:
            continue
        seen_mounts.add(mountpoint)
        percent = (usage.used / usage.total * 100) if usage.total else 0.0
        disks.append(
            DiskInfo(
                device=device,
                mountpoint=mountpoint,
                fstype=fstype,
                total=usage.total,
                used=usage.used,
                free=usage.free,
                percent=round(percent, 1),
            )
        )
    disks.sort(key=lambda item: (item.mountpoint != "/", item.mountpoint))
    if not disks:
        errors.append("Не удалось получить сведения о дисках")
    return disks, errors


def _inode_info(mountpoint: str) -> InodeInfo | None:
    try:
        stats = os.statvfs(mountpoint)
    except OSError:
        return None
    total = stats.f_files
    free = stats.f_ffree
    if total <= 0:
        return None
    used = total - free
    return InodeInfo(
        mountpoint=mountpoint,
        total=total,
        free=free,
        used_percent=round(used / total * 100, 1),
    )


def _btrfs_usage() -> tuple[str, str | None]:
    if not shutil.which("btrfs"):
        return "", "btrfs-progs не установлены"
    result = run_tool(["btrfs", "filesystem", "usage", "-T", "/"], timeout=6)
    if result.ok and result.stdout.strip():
        return result.stdout.strip(), None
    fallback = run_tool(["btrfs", "filesystem", "df", "/"], timeout=6)
    if fallback.ok and fallback.stdout.strip():
        return fallback.stdout.strip(), None
    reason = result.stderr.strip() or fallback.stderr.strip() or result.error or "нет данных"
    if "permission" in reason.lower() or "operation not permitted" in reason.lower():
        return "", "Для подробностей Btrfs нужны права root (используйте ai с подтверждением)"
    return "", reason


def _top_dirs(path: Path, limit: int = 10) -> tuple[list[tuple[str, int]], str | None]:
    if not path.is_dir():
        return [], f"Каталог не найден: {path}"
    result = run_tool(["du", "-x", "-d", "1", "-k", str(path)], timeout=8)
    if not result.ok:
        return [], result.error or result.stderr.strip() or "du недоступен"
    items: list[tuple[str, int]] = []
    for line in result.stdout.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            size_kb = int(parts[0])
        except ValueError:
            continue
        items.append((parts[1].strip(), size_kb * 1024))
    items.sort(key=lambda item: item[1], reverse=True)
    return items[:limit], None


def collect_storage(
    *,
    include_usage: bool = False,
    cwd: Path | None = None,
) -> StorageReport:
    report = StorageReport()
    report.disks, errors = collect_disks()
    report.errors.extend(errors)

    inodes: list[InodeInfo] = []
    for mountpoint in ("/", str(Path.home())):
        info = _inode_info(mountpoint)
        if info and all(existing.mountpoint != info.mountpoint for existing in inodes):
            inodes.append(info)
    report.inodes = inodes

    if report.has_btrfs:
        report.btrfs_usage, report.btrfs_error = _btrfs_usage()

    if include_usage:
        items, error = _top_dirs((cwd or Path.cwd()).resolve())
        report.top_dirs = items
        if error:
            report.errors.append(error)

    return report
