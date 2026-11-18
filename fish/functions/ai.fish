# Ryoku AI Terminal — Fish wrapper.
# Managed by Ryoku AI Terminal installer. Do not edit unless you know what you are doing.
#
# The function intentionally uses `command` so it never calls itself recursively.
# All arguments (including Russian text) are passed through unchanged.
function ai --description 'Ryoku AI Terminal — AI-ассистент для CachyOS/Arch'
    if not command -q ryoku-ai
        echo "ai: команда ryoku-ai не найдена в PATH." >&2
        echo "Установите Ryoku AI Terminal: ./scripts/install.sh" >&2
        return 127
    end

    command ryoku-ai $argv
end
