"""Конфигурация и локальное хранение API-ключей.

Пути по умолчанию:
    ~/.config/ryoku-ai/config.toml
    ~/.config/ryoku-ai/credentials.env
    ~/.local/state/ryoku-ai/history.db

Пути можно переопределить переменными окружения (полезно для тестов):
    RYOKU_AI_CONFIG_DIR, RYOKU_AI_CONFIG_FILE, RYOKU_AI_CREDENTIALS,
    RYOKU_AI_STATE_DIR, RYOKU_AI_HISTORY_DB
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from . import APP_NAME

PROVIDER_NAMES = ("groq", "openrouter")
PROVIDER_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "groq": ("GROQ_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}
_CREDENTIAL_FILE_NAMES = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class ConfigError(Exception):
    """Ошибка чтения или записи конфигурации."""


def _home() -> Path:
    return Path(os.environ.get("HOME", str(Path.home()))).expanduser()


def config_dir() -> Path:
    override = os.environ.get("RYOKU_AI_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else _home() / ".config"
    return base / APP_NAME


def config_path() -> Path:
    override = os.environ.get("RYOKU_AI_CONFIG_FILE")
    if override:
        return Path(override).expanduser()
    return config_dir() / "config.toml"


def credentials_path() -> Path:
    override = os.environ.get("RYOKU_AI_CREDENTIALS")
    if override:
        return Path(override).expanduser()
    return config_dir() / "credentials.env"


def state_dir() -> Path:
    override = os.environ.get("RYOKU_AI_STATE_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg).expanduser() if xdg else _home() / ".local" / "state"
    return base / APP_NAME


def history_db_path() -> Path:
    override = os.environ.get("RYOKU_AI_HISTORY_DB")
    if override:
        return Path(override).expanduser()
    return state_dir() / "history.db"


def atomic_write_text(path: Path, text: str, *, mode: int = 0o600) -> None:
    """Атомарно записать файл с заданными правами."""

    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        path.parent.chmod(0o700)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        os.chmod(path, mode)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        finally:
            raise


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    language: Literal["ru"] = "ru"
    theme: Literal["ryoku", "mono", "accessible"] = "ryoku"

    provider: Literal["auto", "groq", "openrouter"] = "auto"
    provider_order: list[Literal["groq", "openrouter"]] = Field(
        default_factory=lambda: list(PROVIDER_NAMES)
    )
    provider_models: dict[str, str] = Field(default_factory=dict)
    model: str | None = None
    fallback_enabled: bool = True

    request_timeout: float = Field(default=60.0, ge=5.0, le=300.0)
    connect_timeout: float = Field(default=10.0, ge=2.0, le=60.0)
    max_tokens: int = Field(default=2048, ge=128, le=8192)
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)

    command_timeout: float = Field(default=20.0, ge=1.0, le=120.0)
    max_output_chars: int = Field(default=12000, ge=1000, le=200000)
    ai_output_chars: int = Field(default=6000, ge=500, le=50000)
    max_commands: int = Field(default=3, ge=0, le=5)
    confirm_commands: Literal[True] = True
    allow_sudo: bool = True

    history_enabled: bool = True
    history_context_messages: int = Field(default=6, ge=0, le=50)
    max_history: int = Field(default=1000, ge=10, le=100000)
    analyze_results: bool = True
    auto_context: bool = True

    @field_validator("provider_order")
    @classmethod
    def _unique_providers(
        cls, value: list[Literal["groq", "openrouter"]]
    ) -> list[Literal["groq", "openrouter"]]:
        seen: list[Literal["groq", "openrouter"]] = []
        for item in value:
            if item not in seen:
                seen.append(item)
        if not seen:
            raise ValueError("provider_order не может быть пустым")
        return seen

    @field_validator("provider_models")
    @classmethod
    def _clean_models(cls, value: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for provider, model in value.items():
            if provider not in PROVIDER_NAMES:
                continue
            model = model.strip()
            if model and len(model) <= 200:
                cleaned[provider] = model
        return cleaned

    def model_for(self, provider: str) -> str | None:
        return self.provider_models.get(provider)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Неподдерживаемый тип TOML: {type(value)!r}")


def dump_config(config: Config) -> str:
    data = config.model_dump()
    models = data.pop("provider_models", {})
    lines = [
        "# Ryoku AI Terminal — конфигурация",
        "# Секреты здесь не хранятся. API-ключи: credentials.env (chmod 600).",
        "",
    ]
    for key, value in data.items():
        if value is None:
            continue
        lines.append(f"{key} = {_toml_value(value)}")

    lines.append("")
    lines.append("[provider_models]")
    if models:
        for provider, model in sorted(models.items()):
            lines.append(f"{provider} = {_toml_value(model)}")
    else:
        lines.append("# Пример: groq = \"имя-модели\"")
    lines.append("")
    return "\n".join(lines)


def load_config(path: Path | None = None) -> Config:
    path = path or config_path()
    if not path.exists():
        return Config()

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"Не удалось прочитать {path}: {exc}") from exc

    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Некорректный TOML в {path}: {exc}") from exc

    try:
        # Ignore the retired provider in existing config files without requiring edits.
        if data.get("provider") == "gemini":
            data["provider"] = "auto"
        if isinstance(data.get("provider_order"), list):
            original_order = data["provider_order"]
            normalized_order = [
                name for name in data["provider_order"] if name in PROVIDER_NAMES
            ]
            if normalized_order or original_order:
                data["provider_order"] = normalized_order or list(PROVIDER_NAMES)
        if isinstance(data.get("provider_models"), dict):
            data["provider_models"] = {
                name: model
                for name, model in data["provider_models"].items()
                if name in PROVIDER_NAMES
            }
        return Config.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"Некорректная конфигурация {path}: {details}") from exc


def save_config(config: Config, path: Path | None = None) -> Path:
    path = path or config_path()
    atomic_write_text(path, dump_config(config), mode=0o600)
    return path


class Credentials(BaseModel):
    model_config = ConfigDict(extra="ignore")

    groq: str | None = None
    openrouter: str | None = None

    def configured_providers(self) -> list[str]:
        return [name for name in PROVIDER_NAMES if getattr(self, name)]

    def has_any(self) -> bool:
        return bool(self.configured_providers())


class CredentialsError(Exception):
    """Ошибка чтения или записи файла секретов."""


def _clean_secret(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    if not value:
        return ""
    if len(value) > 500 or any(ch.isspace() for ch in value) or "#" in value:
        raise CredentialsError("API-ключ содержит недопустимые символы")
    return value


def parse_credentials(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        try:
            cleaned = _clean_secret(value)
        except CredentialsError:
            continue
        for provider, env_names in PROVIDER_ENV_KEYS.items():
            candidates = {_CREDENTIAL_FILE_NAMES[provider], *env_names}
            if name in candidates and cleaned:
                result[provider] = cleaned
    return result


def load_credentials(path: Path | None = None) -> Credentials:
    path = path or credentials_path()
    values: dict[str, str] = {}
    if path.exists():
        try:
            values.update(parse_credentials(path.read_text(encoding="utf-8")))
        except OSError as exc:
            raise CredentialsError(f"Не удалось прочитать {path}: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise CredentialsError(f"Файл {path} не является UTF-8 текстом") from exc

    for provider, env_names in PROVIDER_ENV_KEYS.items():
        for env_name in env_names:
            env_value = os.environ.get(env_name, "").strip()
            if env_value:
                values[provider] = env_value
                break

    return Credentials(**values)


def save_credentials(credentials: Credentials, path: Path | None = None) -> Path:
    path = path or credentials_path()
    lines = [
        "# Ryoku AI Terminal — локальные API-ключи.",
        "# Файл должен иметь права 600. Не добавляйте его в Git.",
    ]
    for provider in PROVIDER_NAMES:
        value = getattr(credentials, provider)
        if value:
            cleaned = _clean_secret(value)
            if not cleaned:
                continue
            lines.append(f"{_CREDENTIAL_FILE_NAMES[provider]}={cleaned}")
    lines.append("")
    atomic_write_text(path, "\n".join(lines), mode=0o600)
    return path


def credentials_permission_issue(path: Path | None = None) -> str | None:
    """Вернуть предупреждение, если файл секретов доступен другим пользователям."""

    path = path or credentials_path()
    if not path.exists():
        return None
    try:
        mode = path.stat().st_mode & 0o777
    except OSError as exc:
        return f"Не удалось проверить права {path}: {exc}"
    if mode & 0o077:
        return (
            f"Права {path} слишком широкие ({mode:o}). "
            f"Выполните: chmod 600 {path}"
        )
    return None


_SECRET_LIKE = re.compile(r"^[^\s'\"#]{8,500}$")


def validate_secret_format(value: str) -> bool:
    return bool(_SECRET_LIKE.match(value.strip()))
