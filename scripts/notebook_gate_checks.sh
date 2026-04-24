#!/usr/bin/env bash
# Gate checks for the notebook-compression task.
# Outputs GATE_SCORE=N/3 on the last line. Cheap, always-run — catches
# obviously-broken submissions before spending a multi-minute verifier run.
set -uo pipefail

GATE=0
TOTAL=3
DATA_ROOT="${DATA_ROOT:-/mnt/notebook-data}"

# ---------- Gate 1: /app/run exists and is executable ----------
if [ -x /app/run ]; then
    GATE=$((GATE + 1))
    echo "GATE 1 PASS: /app/run exists and is executable"
else
    echo "GATE 1 FAIL: /app/run missing or not executable"
fi

# ---------- Gate 2: visible corpus is populated ----------
if [ -d "${DATA_ROOT}/visible" ] && [ -n "$(ls -A "${DATA_ROOT}/visible" 2>/dev/null)" ]; then
    GATE=$((GATE + 1))
    echo "GATE 2 PASS: visible corpus present at ${DATA_ROOT}/visible"
else
    echo "GATE 2 FAIL: visible corpus missing at ${DATA_ROOT}/visible"
fi

# ---------- Gate 3: python3 + zstandard + nbformat importable ----------
if python3 -c 'import zstandard, nbformat' 2>/dev/null; then
    GATE=$((GATE + 1))
    echo "GATE 3 PASS: python3 zstandard/nbformat available"
else
    echo "GATE 3 FAIL: python3 imports failed"
fi

echo "GATE_SCORE=${GATE}/${TOTAL}"
