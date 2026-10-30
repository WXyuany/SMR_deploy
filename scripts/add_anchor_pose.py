#!/usr/bin/env python3
"""Add reproduced-policy anchor pose fields to a converted motion npz."""
import argparse
from pathlib import Path
import numpy as np
import mujoco

def main():
 p=argparse.ArgumentParser(); p.add_argument('--xml',type=Path,default=Path(__file__).parents[1]/'assets/robot/g1.xml'); p.add_argument('--input',type=Path); p.add_argument('--output',type=Path); p.add_argument('--input-dir',type=Path); p.add_argument('--output-dir',type=Path); a=p.parse_args()
 if a.input_dir:
  a.output_dir.mkdir(parents=True,exist_ok=True)
  for src in sorted(a.input_dir.glob('*.npz')):
   a.input=src; a.output=a.output_dir/src.name; convert(a)
  return
 if not a.input or not a.output: p.error('use --input/--output or --input-dir/--output-dir')
 convert(a)

def convert(a):
 d=np.load(a.input); model=mujoco.MjModel.from_xml_path(str(a.xml)); data=mujoco.MjData(model); aid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'torso_link')
 q=d['joint_pos']; rp=d['root_pos']; rq=d['root_quat']; n=len(q); pos=np.empty((n,3),np.float32); quat=np.empty((n,4),np.float32)
 for i in range(n):
  data.qpos[:3]=rp[i]; data.qpos[3:7]=rq[i]; data.qpos[7:36]=q[i]; mujoco.mj_forward(model,data); pos[i]=data.xpos[aid]; quat[i]=data.xquat[aid]
 out={k:d[k] for k in d.files}; out['anchor_pos_w']=pos; out['anchor_quat_w']=quat; a.output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output,**out); print(a.output,n)
if __name__=='__main__': main()
