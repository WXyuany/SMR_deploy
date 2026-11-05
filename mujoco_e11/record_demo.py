"""Record a connected walk -> standby -> wave MuJoCo demo using the deployment controller."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from controller import A, START, Controller, State


def load_pose(data, motion_dir: Path, name: str, frame: int = 0):
    with np.load(motion_dir / f"{name}.npz") as z:
        data.qpos[:3] = z["root_pos"][frame]
        data.qpos[3:7] = z["root_quat"][frame]
        data.qpos[7:] = z["joint_pos"][frame]
        data.qvel[:3] = z["root_lin_vel"][frame]
        data.qvel[3:6] = z["root_ang_vel_b"][frame]
        data.qvel[6:] = z["joint_vel"][frame]
    data.qfrc_applied[:] = 0
    mujoco.mj_forward(data.model, data)


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0, 180))
    draw.text((12, 9), text, fill=(255, 255, 255, 255))
    return np.asarray(image)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, default=Path(__file__).parent / "assets")
    ap.add_argument("--output", type=Path, default=Path("videos/mujoco_walk_wave_demo.mp4"))
    ap.add_argument("--align-seconds", type=float, default=4.0)
    ap.add_argument("--return-seconds", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    walk = "neutral_walk_ff_180_R_002__A534"
    wave = "wave_two_hands_R_001__A533"
    policy = Controller(args.assets, walk, args.align_seconds, return_seconds=args.return_seconds)

    spec = mujoco.MjSpec.from_file(str(args.assets / "robot" / "g1.xml"))
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.1], rgba=[0.70, 0.70, 0.70, 1.0],
    )
    model = spec.compile()
    model.opt.timestep = 0.005
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    model.dof_armature[6:] = policy.cfg["armature"]
    data = mujoco.MjData(model)
    load_pose(data, args.assets / "motions", walk)
    support_root = data.qpos[:7].copy()
    support_vel = data.qvel[:6].copy()
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    phase = "aligning walk"
    current_target = data.qpos[7:].copy()
    next_control = 0.0
    next_frame = 0.0
    a_sent = False
    align_start_time = 0.0
    switched = False
    transition_started = False
    frame_dt = 1.0 / args.fps
    max_time = 44.0

    with imageio.get_writer(
        args.output, fps=args.fps, codec="libx264", quality=7,
        macro_block_size=1, ffmpeg_log_level="error",
    ) as writer:
        while data.time < max_time:
            if data.time + 1e-9 >= next_control:
                keys = 0
                if data.time < 1e-9:
                    keys = START
                elif not a_sent and data.time >= align_start_time + policy.align_seconds + 0.5:
                    keys = A
                    a_sent = True
                    phase = "walking"

                target = policy.tick(
                    data.time, data.qpos[7:].copy(), data.qvel[6:].copy(),
                    data.qpos[3:7].copy(), data.qvel[3:6].copy(), keys,
                )
                if target is not None:
                    current_target = policy.bounded_target(target, data.qpos[7:], data.qvel[6:])

                # The transition between clips uses the same smooth alignment
                # idea as the operator's s command, without dropping the video.
                if not switched and policy.state == State.STANDBY:
                    switched = True
                    policy.select_motion(wave)
                    policy.align_seconds = 3.0
                    policy.start_q = data.qpos[7:].copy()
                    policy.start_time = data.time
                    align_start_time = data.time
                    policy.state = State.ALIGN
                    policy.reason = "smooth transition to wave first frame"
                    policy.frame = 0
                    support_root = data.qpos[:7].copy()
                    support_vel = data.qvel[:6].copy()
                    a_sent = False
                    phase = "transitioning to wave"
                    transition_started = True
                elif switched and policy.state == State.POLICY and transition_started:
                    phase = "waving"
                    transition_started = False

                next_control += 0.02

            # Four 5 ms MuJoCo steps per 50 Hz policy command.
            data.qfrc_applied[6:] = np.clip(
                policy.kp * (current_target - data.qpos[7:]) - policy.kd * data.qvel[6:],
                -policy.effort, policy.effort,
            )
            mujoco.mj_step(model, data)
            if policy.state in (State.ALIGN, State.READY):
                data.qpos[:7] = support_root
                data.qvel[:6] = support_vel
                mujoco.mj_forward(model, data)

            if data.time + 1e-9 >= next_frame:
                renderer.update_scene(data, camera="tracking")
                frame = renderer.render()
                writer.append_data(caption(frame, f"SMR_deploy  |  {phase}  |  t={data.time:05.1f}s"))
                next_frame += frame_dt

            # Stop after the wave has returned to its initial standby pose.
            if switched and not transition_started and policy.state == State.STANDBY and data.time > 30.0:
                break

    print(f"wrote {args.output} ({data.time:.2f}s); final state={policy.state.value}, frame={policy.frame}, reason={policy.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
