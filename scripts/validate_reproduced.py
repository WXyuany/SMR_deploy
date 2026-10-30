import sys,time
import argparse
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reproduced_controller import ReproducedController, State, START, A, B
assets=Path(__file__).resolve().parents[1]/'assets'
parser=argparse.ArgumentParser(); parser.add_argument('--policy',type=Path,required=True); parser.add_argument('--motion',default='walk1_subject1'); parser.add_argument('--seconds',type=float,default=5.0); args=parser.parse_args()
policy=ReproducedController(assets,str(args.policy),args.motion,8)
spec=mujoco.MjSpec.from_file(str(assets/'robot/g1.xml'))
spec.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[0,0,.1],rgba=[.7,.7,.7,1])
for name,kp,kd,effort in zip(policy.joint_names,policy.kp,policy.kd,policy.effort):
 act=spec.add_actuator(name=f'{name}_policy',target=name); act.trntype=mujoco.mjtTrn.mjTRN_JOINT; act.dyntype=mujoco.mjtDyn.mjDYN_NONE; act.gaintype=mujoco.mjtGain.mjGAIN_FIXED; act.biastype=mujoco.mjtBias.mjBIAS_AFFINE; act.gainprm[0]=float(kp); act.biasprm[1]=-float(kp); act.biasprm[2]=-float(kd); act.inheritrange=0.; act.ctrllimited=False; act.forcelimited=True; act.forcerange[:]=[-float(effort),float(effort)]
m=spec.compile(); m.opt.timestep=.005; m.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST; m.opt.iterations=10; m.opt.ls_iterations=20; m.dof_armature[6:]=policy.cfg['armature']; d=mujoco.MjData(m)
with np.load(assets/'reproduced_motions'/f'{args.motion}.npz') as x:
 d.qpos[:3]=x['root_pos'][0]; d.qpos[3:7]=x['root_quat'][0]; d.qpos[7:36]=x['joint_pos'][0]; d.qvel[:3]=x['root_lin_vel'][0]; d.qvel[3:6]=x['root_ang_vel_b'][0]; d.qvel[6:35]=x['joint_vel'][0]
mujoco.mj_forward(m,d)
root_pose=d.qpos[:7].copy(); root_vel=d.qvel[:6].copy(); q=d.qpos[7:36].copy(); dq=d.qvel[6:35].copy();
# start alignment
policy.set_root(d.qpos[:3],d.qpos[3:7]); policy.tick(0,q,dq,d.qpos[3:7],d.qvel[3:6],START)
# align for 8 s at 50 Hz policy rate, with 4 physics substeps
for k in range(400):
 t=(k+1)*.02
 policy.set_root(d.qpos[:3],d.qpos[3:7]); target=policy.tick(t,d.qpos[7:36].copy(),d.qvel[6:35].copy(),d.qpos[3:7].copy(),d.qvel[3:6].copy(),0)
 if target is not None:
  target=policy.bounded_target(target,d.qpos[7:36],d.qvel[6:35])
  d.ctrl[:]=target
  for _ in range(4):
   mujoco.mj_step(m,d); d.qpos[:7]=root_pose; d.qvel[:6]=root_vel; mujoco.mj_forward(m,d)
print('after align',policy.state.value,'qerr',float(np.max(np.abs(d.qpos[7:36]-policy.positions[0]))))
# A and simulate 5 sec with free root
q=d.qpos[7:36].copy();dq=d.qvel[6:35].copy(); policy.set_root(d.qpos[:3],d.qpos[3:7]); policy.tick(8.2,q,dq,d.qpos[3:7],d.qvel[3:6],A)
max_tilt=0; minz=99; maxz=-99; maxa=0; bad=None
for k in range(int(args.seconds/.02)):
 t=8.2+(k+1)*.02
 policy.set_root(d.qpos[:3],d.qpos[3:7]); target=policy.tick(t,d.qpos[7:36].copy(),d.qvel[6:35].copy(),d.qpos[3:7].copy(),d.qvel[3:6].copy(),0)
 if target is not None:
  d.ctrl[:]=target; maxa=max(maxa,float(np.max(np.abs(policy.previous_action))))
 else: d.ctrl[:]=d.qpos[7:36]
 for _ in range(4): mujoco.mj_step(m,d)
 # tilt from pelvis quaternion: gravity z component, approximate angle
 qw,qx,qy,qz=d.qpos[3:7]; c=1-2*(qx*qx+qy*qy); tilt=np.arccos(np.clip(c,-1,1)); max_tilt=max(max_tilt,float(tilt)); minz=min(minz,float(d.qpos[2])); maxz=max(maxz,float(d.qpos[2]))
 if not np.isfinite(d.qpos).all() or d.qpos[2]<0.25: bad=(k,t,float(d.qpos[2])); break
print('after policy',policy.state.value,'frames',policy.frame,'root_z',float(d.qpos[2]),'zrange',minz,maxz,'max_tilt_deg',max_tilt*180/np.pi,'max_action',maxa,'bad',bad)
