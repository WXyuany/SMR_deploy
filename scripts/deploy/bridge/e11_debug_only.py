#!/usr/bin/env python3
"""Release an active Unitree high-level motion mode before low-level control."""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import g1_common as gc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--domain", type=int, default=0)
    args = ap.parse_args()

    gc.init_dds(args.domain, args.iface)
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

    client = MotionSwitcherClient()
    client.SetTimeout(2.0)
    client.Init()
    code, result = client.CheckMode()
    print(f"CheckMode code={code} result={result}", flush=True)
    if code != 0:
        return 2
    name = result.get("name", "") if isinstance(result, dict) else ""
    if not name:
        print("No high-level motion mode is active; already in debug mode.", flush=True)
        return 0
    code = client.ReleaseMode()
    print(f"ReleaseMode code={code}", flush=True)
    return 0 if code == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
