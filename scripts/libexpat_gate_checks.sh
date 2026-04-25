#!/usr/bin/env bash
# Gate checks for the libexpat-to-x86asm task.
# Outputs GATE_SCORE=N/3 on the last line. Cheap, always-run — catches
# obviously-broken environments before spending a multi-minute verifier run.
set -uo pipefail

GATE=0
TOTAL=3

# ---------- Gate 1: workspace + expat headers present ----------
if [ -d /app/asm-port ] && [ -w /app/asm-port ] && [ -f /app/expat-src/lib/expat.h ]; then
    GATE=$((GATE + 1))
    echo "GATE 1 PASS: /app/asm-port writable, expat.h present"
else
    echo "GATE 1 FAIL: workspace or expat headers missing"
fi

# ---------- Gate 2: nasm + as + ld available ----------
if command -v nasm >/dev/null 2>&1 \
        && command -v as >/dev/null 2>&1 \
        && command -v ld >/dev/null 2>&1; then
    GATE=$((GATE + 1))
    echo "GATE 2 PASS: $(nasm -v 2>&1 | head -1), as and ld present"
else
    echo "GATE 2 FAIL: nasm/as/ld not all on PATH"
fi

# ---------- Gate 3: encrypted toolchain bundles present (verifier infra) ----------
if [ -f /usr/lib/x86_64-linux-gnu/.gcc-bundle.enc ] \
        && [ -f /usr/lib/x86_64-linux-gnu/.libexpat-bundle.enc ]; then
    GATE=$((GATE + 1))
    echo "GATE 3 PASS: encrypted gcc + libexpat bundles staged"
else
    echo "GATE 3 FAIL: encrypted bundles missing — verifier will infrastructure-fail"
fi

echo "GATE_SCORE=${GATE}/${TOTAL}"
