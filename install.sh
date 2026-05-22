#!/usr/bin/env bash
# engram install script — Linux / macOS
# Usage: bash install.sh
set -euo pipefail

INSTALL_DIR="$HOME/.config/opencode"

echo "engram installer"
echo "================"
echo ""

# ── Verify Python ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ and try again." >&2
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python 3.10+ required (found $PY_VERSION)." >&2
    exit 1
fi

echo "Python $PY_VERSION  OK"

# ── Locate install dir ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$SCRIPT_DIR" != "$INSTALL_DIR" ]; then
    echo ""
    echo "Copying files to $INSTALL_DIR ..."
    mkdir -p "$INSTALL_DIR"
    cp -r "$SCRIPT_DIR"/. "$INSTALL_DIR/"
fi

cd "$INSTALL_DIR"

# ── Create venv ────────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "Creating virtualenv at $INSTALL_DIR/.venv ..."
    python3 -m venv .venv
else
    echo "Virtualenv already exists, skipping creation."
fi

# ── Install dependencies ───────────────────────────────────────────────────────
echo "Installing dependencies ..."
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -r requirements.txt --quiet

echo ""
echo "Done."
echo ""
echo "Next steps:"
echo "  1. Open preflight.config.json and adjust retrievalConfidenceThreshold / topN if needed."
echo "  2. Restart opencode — it will discover the plugin automatically."
echo "  3. Verify with:"
echo "       $INSTALL_DIR/.venv/bin/python memory.py retrieve_facts test test \"hello\" 3 0.0"
echo ""
