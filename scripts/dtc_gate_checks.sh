#!/usr/bin/env bash
# Gate checks for the dependent-type-checker task.
# Outputs GATE_SCORE=N/3 on the last line. Cheap, always-run — catches
# obviously-broken submissions before spending a multi-minute verifier run.
set -uo pipefail

GATE=0
TOTAL=3

# ---------- Gate 1: workspace + Cargo.toml present ----------
if [ -d /app/type-checker ] && [ -f /app/type-checker/Cargo.toml ] && [ -f /app/type-checker/src/main.rs ]; then
    GATE=$((GATE + 1))
    echo "GATE 1 PASS: /app/type-checker scaffold present"
else
    echo "GATE 1 FAIL: /app/type-checker scaffold missing or incomplete"
fi

# ---------- Gate 2: rustc + cargo available ----------
if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
    GATE=$((GATE + 1))
    echo "GATE 2 PASS: $(rustc --version), $(cargo --version)"
else
    echo "GATE 2 FAIL: rustc/cargo not on PATH"
fi

# ---------- Gate 3: cargo build --release succeeds (uses sccache if available) ----------
if cd /app/type-checker && cargo build --release --quiet 2>/dev/null; then
    GATE=$((GATE + 1))
    echo "GATE 3 PASS: cargo build --release succeeded"
else
    echo "GATE 3 FAIL: cargo build --release failed"
fi

echo "GATE_SCORE=${GATE}/${TOTAL}"
