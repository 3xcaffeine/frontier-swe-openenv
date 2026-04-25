#!/usr/bin/env bash
# entrypoint.sh — Container entrypoint. Starts the timer daemon, then execs
# whatever command Harbor (or docker run) passes.
#
# Baked into the Docker image. The agent cannot modify this.

# Start the timer daemon without inheriting shell startup hooks. Otherwise a
# bash-based timer launcher can recursively re-trigger itself via BASH_ENV.
FRONTIER_TIMER_BOOTSTRAP=1 env -u BASH_ENV -u ENV /app/timer.sh &

# Harbor/Modal starts the sandbox first and execs commands into it later. Keep
# the sandbox alive if no explicit command is provided.
if [ "$#" -eq 0 ]; then
    exec tail -f /dev/null
fi

# Exec the main command (Harbor's exec() calls, or CMD default)
exec "$@"
