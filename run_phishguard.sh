#!/bin/bash
# -----------------------------------------------------------------------------
# PhishGuard AI (Phishing Email Detector) - Linux / Codespaces Launcher
# Port: 8502
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/Project 2" || exit 1

echo "================================================================="
echo "  Starting PhishGuard AI (Project 2: Phishing Email Detector)"
echo "  Port: 8502"
echo "================================================================="

streamlit run app.py --server.port 8502
