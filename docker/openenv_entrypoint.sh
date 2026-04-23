#!/usr/bin/env bash
set -euo pipefail

# Generate pi models.json from env vars (if agent config is provided)
if [ -n "${FSWE_AGENT_API_URL:-}" ]; then
    mkdir -p /root/.pi/agent
    cat > /root/.pi/agent/models.json <<MODELS_EOF
{
  "providers": {
    "openai-compat": {
      "baseUrl": "${FSWE_AGENT_API_URL}",
      "api": "openai-completions",
      "apiKey": "${FSWE_AGENT_API_KEY:-}",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "${FSWE_AGENT_MODEL:-qwen-3.5-27b}",
          "name": "${FSWE_AGENT_MODEL:-qwen-3.5-27b}",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 131072,
          "maxTokens": 65536
        }
      ]
    }
  }
}
MODELS_EOF
    echo "Generated /root/.pi/agent/models.json for provider=openai-compat model=${FSWE_AGENT_MODEL:-qwen-3.5-27b}"
fi

# Start the task timer (budget countdown from the base workspace)
if [ -x /app/timer.sh ]; then
    FRONTIER_TIMER_BOOTSTRAP=1 env -u BASH_ENV -u ENV /app/timer.sh &
fi

# Start the OpenEnv FastAPI server
cd /opt/openenv
exec uvicorn frontier_swe_env.server.app:app \
    --host 0.0.0.0 --port 8000 --log-level info
