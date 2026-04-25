#!/usr/bin/env bash
# entrypoint.sh — Container entrypoint. Starts the timer daemon, then execs
# whatever command Harbor passes.

FRONTIER_TIMER_BOOTSTRAP=1 env -u BASH_ENV -u ENV /app/timer.sh &
exec "$@"
