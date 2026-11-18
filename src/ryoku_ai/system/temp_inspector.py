"""Read-only, bounded size inspection for temporary and user cache directories."""

from __future__ import annotations

import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ._run import format_bytes

InspectionStatus = Literal["success", "partial_success", "failed"]
_BLOCK_SIZE = 512


@dataclass(slots=True)
class DirectoryInspection:
    path: str
    readable_bytes: int = 0
    scanned_entries: int = 0
    inaccessible_directories: int = 0
    inaccessible_entries: int = 0
    disappeared_entries: int = 0
    duration_ms: int = 0
    status: InspectionStatus = "success"
    warnings: list[str] = field(default_factory=list)
    summary: str = ""

    @property
    def readable_size(self) -> str:
        return format_bytes(self.readable_bytes)

    @property
    def exit_code(self) -> int:
        return 0 if self.status == "success" else 1 if self.status == "partial_success" else 2

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "readable_bytes": self.readable_bytes,
            "readable_size": self.readable_size,
            "scanned_entries": self.scanned_entries,
            "inaccessible_directories": self.inaccessible_directories,
            "inaccessible_entries": self.inaccessible_entries,
            "disappeared_entries": self.disappeared_entries,
            "status": self.status,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "warnings": list(self.warnings),
            "summary": self.summary,
        }


def _allocated_bytes(info: os.stat_result) -> int:
    blocks = getattr(info, "st_blocks", None)
    return max(0, int(blocks) * _BLOCK_SIZE) if blocks is not None else max(0, info.st_size)


def inspect_directory(
    directory: Path | str,
    *,
    max_entries: int = 100_000,
    max_depth: int = 32,
) -> DirectoryInspection:
    """Measure allocated bytes without following symlinks or elevating privileges."""

    root = Path(directory).expanduser()
    started = time.monotonic()
    result = DirectoryInspection(path=str(root))

    def finish() -> DirectoryInspection:
        result.duration_ms = int((time.monotonic() - started) * 1000)
        return result

    try:
        root_info = root.lstat()
    except FileNotFoundError:
        result.status = "failed"
        result.summary = f"Каталог не найден: {root}"
        return finish()
    except PermissionError:
        result.status = "failed"
        result.inaccessible_directories = 1
        result.warnings.append("Нет доступа к каталогу; размер не измерен.")
        result.summary = "Каталог недоступен для чтения."
        return finish()
    except OSError as exc:
        result.status = "failed"
        result.summary = f"Не удалось проверить каталог: {exc.strerror or type(exc).__name__}"
        return finish()

    if stat.S_ISLNK(root_info.st_mode):
        result.status = "failed"
        result.warnings.append("Корневой путь является symlink; переход запрещён.")
        result.summary = "Каталог не сканировался: корневой symlink не разрешён."
        return finish()
    if not stat.S_ISDIR(root_info.st_mode):
        result.status = "failed"
        result.summary = "Указанный путь не является каталогом."
        return finish()

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(root, directory_flags)
        opened_root = os.fstat(root_fd)
    except OSError as exc:
        result.status = "failed"
        result.inaccessible_directories = 1 if isinstance(exc, PermissionError) else 0
        result.summary = f"Не удалось безопасно открыть каталог: {exc.strerror or type(exc).__name__}"
        return finish()
    if (opened_root.st_dev, opened_root.st_ino) != (root_info.st_dev, root_info.st_ino):
        os.close(root_fd)
        result.status = "failed"
        result.summary = "Каталог изменился во время проверки; сканирование отменено."
        return finish()

    result.readable_bytes = _allocated_bytes(opened_root)
    pending: list[tuple[int, Path, int]] = [(root_fd, root, 0)]
    visited_objects: set[tuple[int, int]] = {(opened_root.st_dev, opened_root.st_ino)}
    limit_reached = False

    while pending:
        current_fd, current, depth = pending.pop()
        try:
            with os.scandir(current_fd) as entries:
                while True:
                    try:
                        entry = next(entries)
                    except StopIteration:
                        break
                    except PermissionError:
                        result.inaccessible_directories += 1
                        break
                    except FileNotFoundError:
                        result.disappeared_entries += 1
                        break
                    except OSError:
                        result.disappeared_entries += 1
                        break

                    result.scanned_entries += 1
                    if result.scanned_entries > max_entries:
                        result.scanned_entries = max_entries
                        limit_reached = True
                        break
                    try:
                        entry_info = os.stat(
                            entry.name,
                            dir_fd=current_fd,
                            follow_symlinks=False,
                        )
                    except PermissionError:
                        result.inaccessible_entries += 1
                        continue
                    except FileNotFoundError:
                        result.disappeared_entries += 1
                        continue
                    except OSError:
                        result.disappeared_entries += 1
                        continue

                    identity = (entry_info.st_dev, entry_info.st_ino)
                    if identity in visited_objects:
                        continue
                    visited_objects.add(identity)
                    result.readable_bytes += _allocated_bytes(entry_info)
                    if stat.S_ISDIR(entry_info.st_mode):
                        if depth >= max_depth:
                            limit_reached = True
                            continue
                        try:
                            child_fd = os.open(
                                entry.name,
                                directory_flags,
                                dir_fd=current_fd,
                            )
                            child_info = os.fstat(child_fd)
                        except PermissionError:
                            result.inaccessible_directories += 1
                            continue
                        except FileNotFoundError:
                            result.disappeared_entries += 1
                            continue
                        except OSError:
                            result.disappeared_entries += 1
                            continue
                        if (child_info.st_dev, child_info.st_ino) != identity:
                            os.close(child_fd)
                            result.disappeared_entries += 1
                            continue
                        pending.append((child_fd, current / entry.name, depth + 1))
        except PermissionError:
            result.inaccessible_directories += 1
        except FileNotFoundError:
            result.disappeared_entries += 1
        except OSError:
            result.disappeared_entries += 1
        finally:
            os.close(current_fd)

        if limit_reached and result.scanned_entries >= max_entries:
            break

    for pending_fd, _pending_path, _pending_depth in pending:
        os.close(pending_fd)

    if result.inaccessible_directories:
        result.warnings.append(
            f"Недоступные каталоги: {result.inaccessible_directories}."
        )
    if result.inaccessible_entries:
        result.warnings.append(f"Недоступные элементы: {result.inaccessible_entries}.")
    if result.disappeared_entries:
        result.warnings.append(
            f"Элементы исчезли во время сканирования: {result.disappeared_entries}."
        )
    if limit_reached:
        result.warnings.append(
            f"Достигнут лимит обхода ({result.scanned_entries} элементов или глубина {max_depth})."
        )

    partial = bool(
        result.inaccessible_directories
        or result.inaccessible_entries
        or result.disappeared_entries
        or limit_reached
    )
    result.status = "partial_success" if partial else "success"
    result.summary = f"Доступный размер: {result.readable_size}."
    if partial:
        result.summary += " Результат неполный и является нижней оценкой."
    return finish()


def inspect_standard_temp_directories(
    *,
    home: Path | str | None = None,
    max_entries: int = 100_000,
    max_depth: int = 32,
) -> list[DirectoryInspection]:
    home_path = Path(home).expanduser() if home is not None else Path.home()
    targets = (Path("/tmp"), Path("/var/tmp"), home_path / ".cache")
    return [
        inspect_directory(path, max_entries=max_entries, max_depth=max_depth)
        for path in targets
    ]
