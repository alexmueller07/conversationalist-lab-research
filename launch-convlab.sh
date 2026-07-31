#!/usr/bin/env bash
# Open the convlab desktop application on macOS or Linux.
#
# Sets the app up the first time it runs (virtual environment plus
# dependencies), then starts immediately on every run after that.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "First run: setting up convlab. This downloads about 1.6 GB and takes 15-30 minutes."
    echo
    if ! command -v python3 >/dev/null 2>&1; then
        echo "Python 3.10 or newer is required but was not found."
        echo "  macOS:  brew install python-tk"
        echo "  Debian: sudo apt install python3 python3-venv python3-tk"
        exit 1
    fi
    python3 -m venv .venv
    ./.venv/bin/python -m pip install --upgrade pip
    ./.venv/bin/python -m pip install -e ".[semantic]"
    echo
    echo "Setup complete."
    echo
fi

if ! ./.venv/bin/python -c "import tkinter" >/dev/null 2>&1; then
    echo "Tkinter is missing from this Python install."
    echo "  macOS:  brew install python-tk"
    echo "  Debian: sudo apt install python3-tk"
    exit 1
fi

exec ./.venv/bin/python -m convlab.gui
