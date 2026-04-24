#!/usr/bin/env bash
# entrypoint.sh — Container entrypoint. Starts the timer daemon, then execs
# whatever command Harbor (or docker run) passes.

FRONTIER_TIMER_BOOTSTRAP=1 env -u BASH_ENV -u ENV /app/timer.sh &

if [ "$#" -eq 0 ]; then
    exec tail -f /dev/null
fi

exec "$@"
