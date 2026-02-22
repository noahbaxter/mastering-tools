#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[32m'
RED='\033[31m'
CYAN='\033[36m'
RESET='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
BIN_DIR="$HOME/.local/bin"
COMMANDS=(mastering stems loudness normalize declick mastering-compare)

if ! command -v ffmpeg &>/dev/null; then
    echo -e "${RED}✗${RESET} ffmpeg not found. Install with: brew install ffmpeg"
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo -e "${RED}✗${RESET} No .venv found. Run ${CYAN}venv${RESET} in this directory first."
    exit 1
fi

# Install package into the existing venv
echo -e "Installing ${CYAN}audio-tools${RESET} into venv..."
"$VENV_DIR/bin/pip" install -e "$SCRIPT_DIR" --quiet

# Remove old pipx install if present
if pipx list 2>/dev/null | grep -q audio-tools; then
    echo -e "Removing old pipx install..."
    pipx uninstall audio-tools 2>/dev/null || true
fi

# Symlink entry points to ~/.local/bin
mkdir -p "$BIN_DIR"
for cmd in "${COMMANDS[@]}"; do
    src="$VENV_DIR/bin/$cmd"
    dst="$BIN_DIR/$cmd"
    if [[ -f "$src" ]]; then
        ln -sf "$src" "$dst"
        echo -e "  ${GREEN}✓${RESET} $cmd"
    else
        echo -e "  ${RED}✗${RESET} $cmd not found in venv"
    fi
done

echo -e "${GREEN}✓${RESET} Done. Commands linked to ${CYAN}$BIN_DIR${RESET}"
