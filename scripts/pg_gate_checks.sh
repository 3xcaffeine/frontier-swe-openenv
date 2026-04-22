#!/usr/bin/env bash
# Gate checks for the PostgreSQL wire-adapter task.
# Outputs GATE_SCORE=N/4 on the last line.
set -uo pipefail

WORKSPACE="${APP_DIR:-/app}/postgres-sqlite"
GATE=0
TOTAL=4

cd "$WORKSPACE"

# ---------- Gate 1: Does it compile? ----------
if bash build.sh -Doptimize=ReleaseSafe 2>/dev/null; then
    GATE=$((GATE + 1))
    echo "GATE 1 PASS: build succeeded"
else
    echo "GATE 1 FAIL: build failed"
    echo "GATE_SCORE=${GATE}/${TOTAL}"
    exit 0
fi

# Locate the candidate binary
CANDIDATE=""
if [ -x "$WORKSPACE/zig-out/bin/postgres-sqlite" ]; then
    CANDIDATE="$WORKSPACE/zig-out/bin/postgres-sqlite"
else
    while IFS= read -r f; do
        base="$(basename "$f")"
        case "$base" in *.o|*.a|*.so|*.dll|*.dylib) continue ;; esac
        CANDIDATE="$f"
        break
    done < <(find "$WORKSPACE/zig-out/bin" -maxdepth 1 -type f -perm -111 2>/dev/null | sort)
fi

if [ -z "$CANDIDATE" ]; then
    echo "GATE 2 FAIL: no executable found"
    echo "GATE_SCORE=${GATE}/${TOTAL}"
    exit 0
fi

# ---------- Gate 2: Binary runs without segfault ----------
timeout 2 "$CANDIDATE" --help >/dev/null 2>&1
RC=$?
if [ "$RC" -ne 139 ] && [ "$RC" -ne 134 ]; then
    GATE=$((GATE + 1))
    echo "GATE 2 PASS: binary runs (exit $RC)"
else
    echo "GATE 2 FAIL: binary crashed (exit $RC)"
    echo "GATE_SCORE=${GATE}/${TOTAL}"
    exit 0
fi

# Set up a temp directory with symlinks (same pattern as smoke_test.sh)
TMP=$(mktemp -d)
cleanup() {
    if [ -x "$TMP/bin/pg_ctl" ]; then
        "$TMP/bin/pg_ctl" -D "$TMP/data" -m fast stop >/dev/null 2>&1 || true
    fi
    rm -rf "$TMP"
}
trap cleanup EXIT

mkdir -p "$TMP/bin"
ln -sf "$CANDIDATE" "$TMP/bin/postgres"
ln -sf "$CANDIDATE" "$TMP/bin/initdb"
ln -sf "$CANDIDATE" "$TMP/bin/pg_ctl"
export PATH="$TMP/bin:$PATH"

# ---------- Gate 3: initdb creates a data directory ----------
if timeout 10 initdb -D "$TMP/data" >/dev/null 2>&1 && [ -d "$TMP/data" ]; then
    GATE=$((GATE + 1))
    echo "GATE 3 PASS: initdb created data directory"
else
    echo "GATE 3 FAIL: initdb did not create data directory"
    echo "GATE_SCORE=${GATE}/${TOTAL}"
    exit 0
fi

# ---------- Gate 4: pg_ctl starts a process that listens on TCP ----------
PORT=55444
if timeout 15 pg_ctl -D "$TMP/data" -o "-p $PORT" -w start >/dev/null 2>&1; then
    if timeout 2 bash -c "echo | nc -w1 127.0.0.1 $PORT" >/dev/null 2>&1; then
        GATE=$((GATE + 1))
        echo "GATE 4 PASS: server listening on port $PORT"
    else
        echo "GATE 4 FAIL: server started but not listening on port $PORT"
    fi
    pg_ctl -D "$TMP/data" -m fast stop >/dev/null 2>&1 || true
else
    echo "GATE 4 FAIL: pg_ctl start failed"
fi

echo "GATE_SCORE=${GATE}/${TOTAL}"
