#!/bin/bash
# -----------------------------------------------------------------------------
# TruthGuard AI (Fake News Detector) - Linux / Codespaces Launcher
# Port: 8501
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/Project 1" || exit 1

echo "================================================================="
echo "  Starting TruthGuard AI (Project 1: Fake News Detector)"
echo "  Port: 8501"
echo "================================================================="

streamlit run app.py --server.port 8501
