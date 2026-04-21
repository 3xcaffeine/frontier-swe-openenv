#!/usr/bin/env bash
set -euo pipefail

# Start the task timer (budget countdown from the base workspace)
if [ -x /app/timer.sh ]; then
    FRONTIER_TIMER_BOOTSTRAP=1 env -u BASH_ENV -u ENV /app/timer.sh &
fi

# Start the OpenEnv FastAPI server
cd /opt/openenv
exec uvicorn frontier_swe_env.server.app:app \
    --host 0.0.0.0 --port 8000 --log-level info
