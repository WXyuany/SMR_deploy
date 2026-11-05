#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
IFACE=${IFACE:-eth0}
ART=${ART:-deploy_artifacts/e11c_0220000}
CLIP_DIR=${CLIP_DIR:-deploy_artifacts/clips_e11c}
LOG=${LOG:-deploy_logs/real}
RETURN_SECONDS=${RETURN_SECONDS:-3.0}

# The SDK Python package is only needed for the read-only CheckMode/ReleaseMode
# handoff. Set UNITREE_SDK2_PYTHON if it is installed elsewhere on PC2.
if [[ -n "${UNITREE_SDK2_PYTHON:-}" && -d "$UNITREE_SDK2_PYTHON" ]]; then
  export PYTHONPATH="$UNITREE_SDK2_PYTHON${PYTHONPATH:+:$PYTHONPATH}"
else
  for candidate in "$HOME/unitree_sdk2_python" "$HOME/supermimic_deploy-upstream/unitree_sdk2_python"; do
    if [[ -d "$candidate/unitree_sdk2py" ]]; then
      export PYTHONPATH="$candidate${PYTHONPATH:+:$PYTHONPATH}"
      break
    fi
  done
fi

if [[ ! -x scripts/deploy/bridge/g1_bridge ]]; then
  SDK_ROOT=${UNITREE_SDK2:-$HOME/unitree_sdk2}
  if [[ ! -d "$SDK_ROOT" ]]; then
    echo "g1_bridge is missing; set UNITREE_SDK2 to the Unitree SDK2 directory." >&2
    exit 2
  fi
  bash scripts/deploy/bridge/build.sh "$SDK_ROOT"
fi

echo "Keep the robot on the hoist. Press d to release Unitree high-level mode; q to exit."
IFS= read -r -n 1 -p "debug handoff: " key
echo
case "${key,,}" in
  d) python3 scripts/deploy/bridge/e11_debug_only.py --iface "$IFACE" ;;
  q) exit 0 ;;
  *) echo "Expected d or q." >&2; exit 2 ;;
esac

echo "Starting e11c_0220000 keyboard runner"
echo "clips: $CLIP_DIR/neutral_walk_ff_180_R_002__A534.npz $CLIP_DIR/wave_two_hands_R_001__A533.npz"
echo "Keyboard: s=align, a=standby/replay, x=run, b=DAMP"
exec python3 scripts/deploy/bridge/g1_runner_bridge.py \
  --iface "$IFACE" --domain 0 --stage ground --allow-ssh \
  --artifacts "$ART" \
  --clip "$CLIP_DIR/neutral_walk_ff_180_R_002__A534.npz" \
        "$CLIP_DIR/wave_two_hands_R_001__A533.npz" \
  --return-seconds "$RETURN_SECONDS" --log-dir "$LOG" \
  --thermal-stop-temp 78 --max-temp 90 --warn-temp 75
