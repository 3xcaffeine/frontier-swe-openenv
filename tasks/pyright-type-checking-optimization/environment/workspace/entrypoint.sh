#!/usr/bin/env bash
# entrypoint.sh — Container entrypoint. Starts the timer daemon, runs the
# agent command, then keeps the container alive for the verifier phase.

FRONTIER_TIMER_BOOTSTRAP=1 env -u BASH_ENV -u ENV /app/timer.sh &
exec "$@"
