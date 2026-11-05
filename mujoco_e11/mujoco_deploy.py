"""Interactive CPU MuJoCo deployment loop using the same ONNX controller as G1."""

import argparse
import queue
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from controller import START, A, B, Controller, State
from mujoco.glfw import glfw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path(__file__).parent / "assets")
    parser.add_argument("--motion", default="walk1_subject1")
    parser.add_argument("--align-seconds", type=float, default=3.0)
    parser.add_argument("--return-seconds", type=float, default=3.0,
                        help="policy-gain smooth return to the initial standby pose after each clip")
    args = parser.parse_args()
    policy = Controller(args.assets, args.motion, args.align_seconds,
                        return_seconds=args.return_seconds)

    spec = mujoco.MjSpec.from_file(str(args.assets / "robot" / "g1.xml"))
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.1], rgba=[0.7, 0.7, 0.7, 1],
    )
    model = spec.compile()
    model.opt.timestep = 0.005
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    model.dof_armature[6:] = policy.cfg["armature"]
    data = mujoco.MjData(model)
    events = queue.SimpleQueue()
    root_pose = np.zeros(7, dtype=np.float64)
    root_vel = np.zeros(6, dtype=np.float64)

    def reset_pose():
        with np.load(args.assets / "motions" / f"{policy.motion}.npz") as motion:
            data.qpos[:3] = motion["root_pos"][0]
            data.qvel[:3] = motion["root_lin_vel"][0]
            data.qvel[3:6] = motion["root_ang_vel_b"][0]
            data.qvel[6:] = motion["joint_vel"][0]
        data.qpos[3:7] = policy.ref_quat[0]
        data.qpos[7:] = policy.positions[0]
        data.qfrc_applied[:] = 0
        mujoco.mj_forward(model, data)
        root_pose[:] = data.qpos[:7]
        root_vel[:] = data.qvel[:6]
        policy.state = State.READY
        policy.reason = "simulation reset to first frame"
        policy.frame = 0
        policy.history.fill(0)
        policy.previous_action.fill(0)
        policy.last_keys = 0

    def key_callback(keycode):
        events.put(keycode)

    reset_pose()
    print("s: align; a: run/replay; b: damp; x/y: previous/next motion")
    print(f"clip end: policy-gain smooth return to initial standby pose over {args.return_seconds:.1f}s")
    print(f"selected: {policy.motion}; iteration: {policy.cfg['iteration']}")
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            cycle = time.monotonic()
            keys = 0
            while not events.empty():
                key = events.get_nowait()
                if key in (glfw.KEY_X, glfw.KEY_Y, glfw.KEY_LEFT_BRACKET, glfw.KEY_RIGHT_BRACKET):
                    index = policy.motions.index(policy.motion)
                    direction = -1 if key in (glfw.KEY_X, glfw.KEY_LEFT_BRACKET) else 1
                    policy.select_motion(policy.motions[(index + direction) % len(policy.motions)])
                    reset_pose()
                    print(f"selected: {policy.motion}; press Enter to run")
                elif key in (glfw.KEY_S, glfw.KEY_SPACE):
                    keys |= START
                elif key in (glfw.KEY_A, glfw.KEY_ENTER):
                    keys |= A
                elif key in (glfw.KEY_B, glfw.KEY_BACKSPACE):
                    keys |= B

            target = policy.tick(
                data.time,
                data.qpos[7:].copy(),
                data.qvel[6:].copy(),
                data.qpos[3:7].copy(),
                data.qvel[3:6].copy(),
                keys,
            )
            # During alignment, emulate the support/harness used for the initial
            # pose check by holding the floating base fixed.  READY is frozen;
            # only POLICY releases the base so the policy can balance it.
            if target is not None and policy.state in (State.ALIGN, State.POLICY, State.RETURN, State.STANDBY):
                target = policy.bounded_target(target, data.qpos[7:], data.qvel[6:])
                for _ in range(4):
                    data.qfrc_applied[6:] = np.clip(
                        policy.kp * (target - data.qpos[7:]) - policy.kd * data.qvel[6:],
                        -policy.effort, policy.effort,
                    )
                    mujoco.mj_step(model, data)
                    if policy.state == State.ALIGN:
                        data.qpos[:7] = root_pose
                        data.qvel[:6] = root_vel
                        mujoco.mj_forward(model, data)
            # Damping leaves the model passive; press s or select another motion
            # to reset and restore the supported first-frame pose.
            viewer.sync()
            time.sleep(max(0, 0.02 - (time.monotonic() - cycle)))


if __name__ == "__main__":
    main()
