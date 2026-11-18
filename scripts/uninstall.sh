#!/usr/bin/env bash
#
# Ryoku AI Terminal — удаление для CachyOS/Arch Linux.
#
# По умолчанию удаляются только программа, launcher и файлы Fish, созданные
# установщиком. Конфигурация, API-ключи и история сохраняются.
# Для полного удаления используйте --purge.
set -euo pipefail

HOME_DIR="${HOME:?HOME не задан}"

BIN_DIR="${XDG_BIN_HOME:-$HOME_DIR/.local/bin}"
VENV_DIR="${RYOKU_AI_VENV:-${XDG_DATA_HOME:-$HOME_DIR/.local/share}/ryoku-ai/venv}"
FISH_CONFIG="${XDG_CONFIG_HOME:-$HOME_DIR/.config}/fish"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME_DIR/.config}/ryoku-ai"
STATE_DIR="${XDG_STATE_HOME:-$HOME_DIR/.local/state}/ryoku-ai"
ASSUME_YES=0
PURGE=0
FORCE=0

MARKER="Ryoku AI Terminal"

usage() {
    cat <<'EOF'
Удаление Ryoku AI Terminal (CachyOS / Arch Linux).

Использование:
  ./scripts/uninstall.sh [опции]

Опции:
  --bin-dir DIR       каталог launcher-а ryoku-ai (по умолчанию ~/.local/bin)
  --venv DIR          каталог виртуального окружения
  --fish-config DIR   каталог Fish (по умолчанию ~/.config/fish)
  --yes               не задавать вопросов
  --purge             удалить также config.toml, credentials.env и history.db
  --force             удалить launcher и Fish-файлы даже без маркера установщика
  -h, --help          эта справка

По умолчанию пользовательские данные сохраняются:
  ~/.config/ryoku-ai/config.toml
  ~/.config/ryoku-ai/credentials.env
  ~/.local/state/ryoku-ai/history.db

Системный Python и пакеты системы не изменяются.
EOF
}

log() { printf '%s\n' "$*"; }
warn() { printf 'Предупреждение: %s\n' "$*" >&2; }
die() { printf 'Ошибка: %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
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
        --yes)
            ASSUME_YES=1
            shift
            ;;
        --purge)
            PURGE=1
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

if [ "$ASSUME_YES" -ne 1 ]; then
    [ -t 0 ] || die "нет интерактивного терминала. Повторите с --yes."
    printf 'Удалить Ryoku AI Terminal'
    if [ "$PURGE" -eq 1 ]; then
        printf ' ВМЕСТЕ с конфигурацией, ключами и историей'
    fi
    printf '? [y/N] '
    read -r answer
    case "$answer" in
        y|Y|yes|YES|да|Да) ;;
        *) log "Отменено."; exit 0 ;;
    esac
fi

LAUNCHER="$BIN_DIR/ryoku-ai"
VENV_LAUNCHER="$VENV_DIR/bin/ryoku-ai"

# --- launcher --------------------------------------------------------------

if [ -L "$LAUNCHER" ]; then
    target="$(readlink "$LAUNCHER" || true)"
    target_real="$(readlink -f "$LAUNCHER" 2>/dev/null || true)"
    expected_real="$(readlink -f "$VENV_LAUNCHER" 2>/dev/null || true)"
    if [ "$target" = "$VENV_LAUNCHER" ] || { [ -n "$target_real" ] && [ "$target_real" = "$expected_real" ]; }; then
        rm -f -- "$LAUNCHER"
        log "Удалён launcher: $LAUNCHER"
    elif [ "$FORCE" -eq 1 ]; then
        warn "launcher указывает на $target; удаляю из-за --force"
        rm -f -- "$LAUNCHER"
        log "Удалён launcher: $LAUNCHER"
    else
        warn "$LAUNCHER указывает не на Ryoku AI Terminal ($target). Оставляю. Используйте --force."
    fi
elif [ -e "$LAUNCHER" ]; then
    if [ "$FORCE" -eq 1 ] || grep -q "$MARKER\|ryoku_ai" "$LAUNCHER" 2>/dev/null; then
        rm -f -- "$LAUNCHER"
        log "Удалён launcher: $LAUNCHER"
    else
        warn "$LAUNCHER не похож на launcher Ryoku AI Terminal. Оставляю. Используйте --force."
    fi
else
    log "Launcher не найден: $LAUNCHER"
fi

# --- venv ----------------------------------------------------------------

case "$VENV_DIR" in
    /|"$HOME_DIR"|"$HOME_DIR/"|"$HOME_DIR/.local"|"$HOME_DIR/.local/share")
        die "отказ от удаления опасного пути venv: $VENV_DIR"
        ;;
esac

if [ -d "$VENV_DIR" ]; then
    if [ -f "$VENV_DIR/pyvenv.cfg" ] || [ -x "$VENV_DIR/bin/python" ]; then
        rm -rf -- "$VENV_DIR"
        log "Удалено виртуальное окружение: $VENV_DIR"
        rmdir "$(dirname "$VENV_DIR")" 2>/dev/null || true
    elif [ "$FORCE" -eq 1 ]; then
        warn "$VENV_DIR не похож на venv; удаляю из-за --force"
        rm -rf -- "$VENV_DIR"
        log "Удалено: $VENV_DIR"
    else
        warn "$VENV_DIR не похож на venv. Оставляю. Используйте --force."
    fi
else
    log "Виртуальное окружение не найдено: $VENV_DIR"
fi

# --- Fish -----------------------------------------------------------------

FISH_FUNCTION="$FISH_CONFIG/functions/ai.fish"
FISH_CONF="$FISH_CONFIG/conf.d/ryoku-ai.fish"

remove_fish_file() {
    local path="$1"
    local label="$2"
    if [ ! -e "$path" ]; then
        log "Файл не найден ($label): $path"
        return
    fi
    if grep -q "$MARKER" "$path" 2>/dev/null; then
        rm -f -- "$path"
        log "Удалён файл ($label): $path"
    elif [ "$FORCE" -eq 1 ]; then
        warn "$path не содержит маркер установщика; удаляю из-за --force"
        rm -f -- "$path"
    else
        warn "$path не принадлежит Ryoku AI Terminal. Оставляю. Используйте --force."
    fi
}

remove_fish_file "$FISH_FUNCTION" "Fish-функция ai"
remove_fish_file "$FISH_CONF" "Fish conf.d"

# --- пользовательские данные ---------------------------------------------

if [ "$PURGE" -eq 1 ]; then
    for path in "$CONFIG_DIR" "$STATE_DIR"; do
        case "$path" in
            /|"$HOME_DIR"|"$HOME_DIR/"|"$HOME_DIR/.config"|"$HOME_DIR/.local"|"$HOME_DIR/.local/state")
                die "отказ от удаления опасного пути: $path"
                ;;
        esac
        if [ -e "$path" ]; then
            rm -rf -- "$path"
            log "Удалено: $path"
        else
            log "Не найдено: $path"
        fi
    done
    log "Конфигурация, API-ключи и история удалены."
else
    log ""
    log "Пользовательские данные сохранены:"
    log "  $CONFIG_DIR/config.toml"
    log "  $CONFIG_DIR/credentials.env"
    log "  $STATE_DIR/history.db"
    log "Для полного удаления: ./scripts/uninstall.sh --purge --yes"
fi

log ""
log "Готово. Откройте новый терминал или выполните: exec fish"
