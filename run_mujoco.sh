#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
exec "$PYTHON" "$ROOT/reproduced_mujoco.py" --policy "$ROOT/assets/reproduced_policy_lafan1_anchor_5080.onnx" "$@"
