"""MuJoCo validation runner for a SuperMimic-Reproduced 689-dim policy."""
import argparse
import queue
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from mujoco.glfw import glfw

from reproduced_controller import A, B, START, X, Y, ReproducedController, State


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, default=Path(__file__).parent / 'assets')
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--motion', default='walk1_subject1')
    p.add_argument('--align-seconds', type=float, default=8.0)
    p.add_argument('--skip-start-pose-check', action='store_true')
    a = p.parse_args()
    policy = ReproducedController(a.assets, a.policy, a.motion, a.align_seconds,
                                  skip_start_pose_check=a.skip_start_pose_check)

    spec = mujoco.MjSpec.from_file(str(a.assets / 'robot' / 'g1.xml'))
    spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, 0.1], rgba=[0.7, 0.7, 0.7, 1])
    # Match mjlab's BuiltinPositionActuatorCfg exactly.  The training runner
    # uses native MuJoCo position actuators (implicit PD), rather than an
    # explicit qfrc_applied PD loop; the latter is unstable at these gains.
    for name, kp, kd, effort in zip(policy.joint_names, policy.kp, policy.kd, policy.effort):
        act = spec.add_actuator(name=f'{name}_policy', target=name)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.dyntype = mujoco.mjtDyn.mjDYN_NONE
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = float(kp)
        act.biasprm[1] = -float(kp)
        act.biasprm[2] = -float(kd)
        act.inheritrange = 0.0
        act.ctrllimited = False
        act.forcelimited = True
        act.forcerange[:] = [-float(effort), float(effort)]
    model = spec.compile()
    model.opt.timestep = 0.005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    model.dof_armature[6:] = policy.cfg['armature']
    data = mujoco.MjData(model)
    events = queue.SimpleQueue()
    root_pose = np.zeros(7, dtype=np.float64)
    root_vel = np.zeros(6, dtype=np.float64)

    def reset_pose():
        motion_file = policy.motion_dir / f'{policy.motion}.npz'
        with np.load(motion_file) as m:
            data.qpos[:3] = m['root_pos'][0]
            data.qpos[3:7] = m['root_quat'][0]
            data.qpos[7:36] = m['joint_pos'][0]
            data.qvel[:3] = m['root_lin_vel'][0]
            data.qvel[3:6] = m['root_ang_vel_b'][0]
            data.qvel[6:35] = m['joint_vel'][0]
        data.qfrc_applied[:] = 0
        mujoco.mj_forward(model, data)
        root_pose[:] = data.qpos[:7]
        root_vel[:] = data.qvel[:6]
        policy.state = State.READY
        policy.reason = 'simulation reset to first frame'
        policy.frame = 0
        policy.history.fill(0)
        policy.previous_action.fill(0)
        policy.last_keys = 0

    reset_pose()
    print('s: align; a: run; b: damp; x/y: previous/next motion')
    print(f'selected: {policy.motion}; goal_dim={policy.goal_dim}; action_dim=29')
    with mujoco.viewer.launch_passive(model, data, key_callback=events.put) as viewer:
        while viewer.is_running():
            cycle = time.monotonic()
            keys = 0
            while not events.empty():
                key = events.get_nowait()
                if key in (glfw.KEY_X, glfw.KEY_Y):
                    i = policy.motions.index(policy.motion)
                    policy.select_motion(policy.motions[(i + (-1 if key == glfw.KEY_X else 1)) % len(policy.motions)])
                    reset_pose()
                    print(f'selected: {policy.motion}')
                elif key == glfw.KEY_S:
                    keys |= START
                elif key == glfw.KEY_A:
                    keys |= A
                elif key == glfw.KEY_B:
                    keys |= B

            policy.set_root(data.qpos[:3], data.qpos[3:7])
            target = policy.tick(data.time, data.qpos[7:36].copy(), data.qvel[6:35].copy(),
                                 data.qpos[3:7].copy(), data.qvel[3:6].copy(), keys)
            if target is not None and policy.state in (State.ALIGN, State.POLICY):
                target = policy.bounded_target(target, data.qpos[7:36], data.qvel[6:35])
                for _ in range(4):
                    data.ctrl[:] = target
                    mujoco.mj_step(model, data)
                    if policy.state == State.ALIGN:
                        data.qpos[:7] = root_pose
                        data.qvel[:6] = root_vel
                        mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(max(0, 0.02 - (time.monotonic() - cycle)))


if __name__ == '__main__':
    main()
