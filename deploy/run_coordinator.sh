#!/usr/bin/env bash
# Deploy + run the Centurion DiLoCo coordinator on a cloud server (e.g. GCP).
#
# This lets Windows and Mac workers collaborate over the public internet,
# bypassing campus/LAN client-isolation. It runs alongside the teammate's
# pipeline server (which uses port 9998) by using a DIFFERENT port (9100).
#
# Usage on the cloud box (from the repo root, after `git clone` + `git checkout`):
#   bash deploy/run_coordinator.sh
#
# Override defaults via env vars, e.g.:
#   PORT=9100 WORLD_SIZE=2 ROUNDS=20 bash deploy/run_coordinator.sh

set -euo pipefail

PORT="${PORT:-9100}"
WORLD_SIZE="${WORLD_SIZE:-2}"
ROUNDS="${ROUNDS:-20}"
HOST="${HOST:-0.0.0.0}"      # 0.0.0.0 = listen on all interfaces (public)

cd "$(dirname "$0")/.."       # repo root

# 1. Create an isolated venv if missing
if [ ! -d ".venv-coord" ]; then
  echo "[deploy] creating venv .venv-coord"
  python3 -m venv .venv-coord
fi

# 2. Install the tiny coordinator-only dependency set
echo "[deploy] installing coordinator deps (fastapi/uvicorn/numpy/safetensors)"
./.venv-coord/bin/python -m pip install --upgrade pip -q
./.venv-coord/bin/python -m pip install -q -r deploy/requirements-coord.txt

# 3. Run the coordinator
echo "[deploy] starting coordinator on ${HOST}:${PORT} world_size=${WORLD_SIZE} rounds=${ROUNDS}"
echo "[deploy] workers connect with:  --coord-url http://<PUBLIC_IP>:${PORT}"
exec ./.venv-coord/bin/python -m centurion_coord.server \
  --host "${HOST}" --port "${PORT}" \
  --world-size "${WORLD_SIZE}" --rounds "${ROUNDS}"
