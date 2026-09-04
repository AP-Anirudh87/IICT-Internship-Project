#!/bin/bash
# -----------------------------------------------------------------------------
# IICT Internship Project - Automated Linux / Codespaces Environment Setup
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "================================================================="
echo "  Setting up IICT Internship Project Environment (Linux / Codespaces)"
echo "================================================================="

if [ ! -d ".venv" ]; then
    echo "[1/3] Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "[INFO] .venv already exists."
fi

echo "[2/3] Upgrading pip..."
source .venv/bin/activate
pip install --upgrade pip --quiet

echo "[3/3] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "================================================================="
echo "  SUCCESS! Environment setup is complete."
echo "  Launch apps with:"
echo "    bash run_truthguard.sh"
echo "    bash run_phishguard.sh"
echo "================================================================="
