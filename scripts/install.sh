#!/usr/bin/env bash
#
# Ryoku AI Terminal — установка для CachyOS/Arch Linux.
# Скрипт никогда не устанавливает пакеты в системный Python: только в venv.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="${HOME:?HOME не задан}"

BIN_DIR="${XDG_BIN_HOME:-$HOME_DIR/.local/bin}"
VENV_DIR="${RYOKU_AI_VENV:-${XDG_DATA_HOME:-$HOME_DIR/.local/share}/ryoku-ai/venv}"
FISH_CONFIG="${XDG_CONFIG_HOME:-$HOME_DIR/.config}/fish"
INSTALL_FISH=1
OFFLINE=0
FORCE=0
PYTHON_BIN="${PYTHON:-python3}"

usage() {
    cat <<'EOF'
Установка Ryoku AI Terminal (CachyOS / Arch Linux).

Использование:
  ./scripts/install.sh [опции]

Опции:
  --prefix DIR        установить launcher в DIR/bin
  --bin-dir DIR       каталог для ryoku-ai (по умолчанию ~/.local/bin)
  --venv DIR          каталог виртуального окружения
  --fish-config DIR   каталог Fish (по умолчанию ~/.config/fish)
  --no-fish           не устанавливать Fish-функцию ai
  --offline           не обращаться к PyPI; проверить уже установленные зависимости
  --force             продолжить на не-Arch системе / перезаписать чужой ai.fish
  -h, --help          эта справка

Зависимости ставятся только в venv через pip. Системный Python не изменяется.
Пользовательские config.toml, credentials.env и history.db при удалении
программы по умолчанию сохраняются.
EOF
}

log() { printf '%s\n' "$*"; }
warn() { printf 'Предупреждение: %s\n' "$*" >&2; }
die() { printf 'Ошибка: %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix)
            [ $# -ge 2 ] || die "--prefix требует каталог"
            BIN_DIR="$2/bin"
            shift 2
            ;;
        --bin-dir)
            [ $# -ge 2 ] || die "--bin-dir требует каталог"
            BIN_DIR="$2"
            shift 2
            ;;
        --venv)
            [ $# -ge 2 ] || die "--venv требует каталог"
            VENV_DIR="$2"
            shift 2
            ;;
        --fish-config)
            [ $# -ge 2 ] || die "--fish-config требует каталог"
            FISH_CONFIG="$2"
            shift 2
            ;;
        --no-fish)
            INSTALL_FISH=0
            shift
            ;;
        --offline)
            OFFLINE=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "неизвестная опция: $1 (см. --help)"
            ;;
    esac
done

[ -d "$ROOT_DIR/src/ryoku_ai" ] || die "не найден src/ryoku_ai в $ROOT_DIR"
[ -f "$ROOT_DIR/pyproject.toml" ] || die "не найден pyproject.toml в $ROOT_DIR"

if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
fi
DISTRO_TEXT="${ID:-} ${ID_LIKE:-} ${PRETTY_NAME:-}"
case "$DISTRO_TEXT" in
    *arch*|*cachyos*|*manjaro*|*endeavouros*)
        ;;
    *)
        if [ "$FORCE" -ne 1 ]; then
            die "система не похожа на Arch/CachyOS (${PRETTY_NAME:-неизвестно}). Используйте --force только осознанно."
        fi
        warn "продолжаю на не-Arch системе из-за --force"
        ;;
esac

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "не найден $PYTHON_BIN"
"$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
    || die "требуется Python 3.10 или новее"

# Не трогаем чужие Fish-файлы. Проверяем до создания venv, чтобы установка
# не осталась наполовину выполненной.
if [ "$INSTALL_FISH" -eq 1 ] && command -v fish >/dev/null 2>&1; then
    for candidate in \
        "$FISH_CONFIG/functions/ai.fish" \
        "$FISH_CONFIG/conf.d/ryoku-ai.fish"
    do
        if [ -e "$candidate" ] && ! grep -q "Ryoku AI Terminal" "$candidate" 2>/dev/null; then
            if [ "$FORCE" -ne 1 ]; then
                die "уже существует чужой $candidate. Не перезаписываю без --force."
            fi
            warn "перезапишу $candidate из-за --force"
        fi
    done
fi

log "Каталог проекта: $ROOT_DIR"
log "Виртуальное окружение: $VENV_DIR"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    log "Создаю venv…"
    "$PYTHON_BIN" -m venv "$VENV_DIR" || die "не удалось создать venv"
fi

VENV_PY="$VENV_DIR/bin/python"
[ -x "$VENV_PY" ] || die "в venv нет $VENV_PY"

if [ "$OFFLINE" -eq 1 ]; then
    log "Режим --offline: проверяю зависимости без PyPI…"
    "$VENV_PY" - <<'PY' || die "в venv не хватает зависимостей; повторите без --offline"
import sys
import typer  # noqa: F401
import rich  # noqa: F401
import httpx  # noqa: F401
import pydantic  # noqa: F401
if sys.version_info < (3, 11):
    import tomli  # noqa: F401
PY
else
    log "Устанавливаю зависимости в venv…"
    "$VENV_PY" -m pip install --disable-pip-version-check \
        "typer>=0.12.0" "rich>=13.7.0" "httpx>=0.27.0" "pydantic>=2.6.0" \
        "tomli>=2.0; python_version < '3.11'" \
        || die "pip не смог установить зависимости (проверьте сеть или используйте --offline)"
fi

SITE_PACKAGES="$("$VENV_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
[ -n "$SITE_PACKAGES" ] || die "не удалось определить site-packages"

log "Копирую пакет в venv…"
rm -rf -- "$SITE_PACKAGES/ryoku_ai"
cp -a "$ROOT_DIR/src/ryoku_ai" "$SITE_PACKAGES/ryoku_ai"
"$VENV_PY" -m compileall -q "$SITE_PACKAGES/ryoku_ai" || true

cat > "$VENV_DIR/bin/ryoku-ai" <<EOF
#!/bin/sh
exec "$VENV_PY" -m ryoku_ai "\$@"
EOF
chmod 755 "$VENV_DIR/bin/ryoku-ai"

mkdir -p "$BIN_DIR"
ln -sfn "$VENV_DIR/bin/ryoku-ai" "$BIN_DIR/ryoku-ai"
log "Launcher: $BIN_DIR/ryoku-ai"

"$VENV_PY" - <<'PY' || die "установленный пакет не импортируется"
from ryoku_ai.cli import app  # noqa: F401
from ryoku_ai import __version__
print(f"Ryoku AI Terminal {__version__} установлен")
PY

if [ "$INSTALL_FISH" -eq 1 ]; then
    if command -v fish >/dev/null 2>&1; then
        FUNCTIONS_DIR="$FISH_CONFIG/functions"
        CONF_DIR="$FISH_CONFIG/conf.d"
        TARGET="$FUNCTIONS_DIR/ai.fish"
        CONF_TARGET="$CONF_DIR/ryoku-ai.fish"
        mkdir -p "$FUNCTIONS_DIR" "$CONF_DIR"

        if [ -e "$TARGET" ] && ! grep -q "Ryoku AI Terminal" "$TARGET" 2>/dev/null; then
            if [ "$FORCE" -ne 1 ]; then
                die "уже существует чужая функция $TARGET. Не перезаписываю без --force."
            fi
            warn "перезаписываю $TARGET из-за --force"
        fi
        install -m 644 "$ROOT_DIR/fish/functions/ai.fish" "$TARGET"
        log "Fish-функция: $TARGET"

        if [ -e "$CONF_TARGET" ] && ! grep -q "Ryoku AI Terminal" "$CONF_TARGET" 2>/dev/null; then
            if [ "$FORCE" -ne 1 ]; then
                die "уже существует чужой $CONF_TARGET. Не перезаписываю без --force."
            fi
        fi
        {
            printf '%s\n' '# Ryoku AI Terminal — добавляет launcher в PATH текущей сессии Fish.'
            printf '%s\n' '# Управляется установщиком Ryoku AI Terminal.'
            printf 'if not contains -- "%s" $PATH\n' "$BIN_DIR"
            printf '    set -gx PATH "%s" $PATH\n' "$BIN_DIR"
            printf '%s\n' 'end'
        } > "$CONF_TARGET"
        chmod 644 "$CONF_TARGET"
        log "Fish conf.d: $CONF_TARGET"
    else
        warn "fish не найден: функция ai не установлена. Установите Fish и повторите."
    fi
fi

if ! printf '%s' ":$PATH:" | grep -q ":$BIN_DIR:"; then
    warn "$BIN_DIR отсутствует в PATH. Fish получит путь через conf.d, для других оболочек добавьте каталог в PATH."
fi

log ""
log "Готово. Проверка:"
log "  $BIN_DIR/ryoku-ai --version"
log "  ryoku-ai setup"
log "  ai почему медленный интернет"
log ""
log "API-ключи: ~/.config/ryoku-ai/credentials.env (chmod 600)"
log "Конфигурация и история не удаляются при деинсталляции по умолчанию."
