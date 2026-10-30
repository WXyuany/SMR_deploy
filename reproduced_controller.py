"""Controller for SuperMimic-Reproduced policy checkpoints exported to ONNX."""
from __future__ import annotations
import enum, json, time, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import onnxruntime as ort

START,A,B=1<<2,1<<8,1<<9
X,Y=1<<10,1<<11

def qnorm(q):
 q=np.asarray(q,np.float32); return q/max(float(np.linalg.norm(q)),1e-8)
def qmul(a,b):
 aw,ax,ay,az=a; bw,bx,by,bz=b
 return np.array([aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw],np.float32)
def qinv(q):
 q=qnorm(q); return np.array([q[0],-q[1],-q[2],-q[3]],np.float32)
def qapply(q,v): return qmul(qmul(q,np.r_[0,np.asarray(v,np.float32)]),qinv(q))[1:]
def qaxis(axis,angle):
 axis=np.asarray(axis,np.float32); axis/=max(float(np.linalg.norm(axis)),1e-8); h=.5*angle
 return np.r_[np.cos(h),axis*np.sin(h)].astype(np.float32)
def qyaw(q):
 w,x,y,z=qnorm(q); return qnorm(np.array([w,0,0,z],np.float32))
def qmat(q):
 w,x,y,z=qnorm(q)
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],np.float32)
def gravity(q):
 # projected_gravity_b: world gravity expressed in the body frame.
 # q is the body-to-world MuJoCo quaternion, so use its inverse.
 return qapply(qinv(q),[0,0,-1])
def rot6(q): return qmat(q)[:,:2].reshape(6)
def flatten_history(h):
 return np.concatenate([h[:,lo:hi].reshape(-1) for lo,hi in ((0,29),(29,58),(58,61),(61,64),(64,93))])[None].astype(np.float32)

class AnchorFK:
 def __init__(self,xml,joint_names):
  root=ET.parse(xml).getroot(); self.nodes={}; self.jidx={n:i for i,n in enumerate(joint_names)}; self._walk(root.find('./worldbody'),None)
 def _vec(self,s,n,default):
  x=s.get(n); return np.fromstring(x,sep=' ',dtype=np.float32) if x else np.asarray(default,np.float32)
 def _walk(self,parent,pid):
  kids=list(parent)
  for b in kids:
   if b.tag!='body': continue
   name=b.get('name','')
   # MuJoCo's free-jointed root body pose is supplied by qpos[0:7].  Its
   # XML `pos` is the model's nominal body placement and must not be added
   # again to a runtime root pose (pelvis has pos z=0.793 in g1.xml).
   is_root = pid is None
   node={'parent':('__root__' if is_root else pid),
         'pos':np.zeros(3,np.float32) if is_root else self._vec(b,'pos',[0,0,0]),
         'quat':np.array([1,0,0,0],np.float32) if is_root else qnorm(self._vec(b,'quat',[1,0,0,0])),
         'joint':None}
   j=b.find('joint')
   if j is not None and j.get('type','hinge')=='hinge':
    node['joint']={'name':j.get('name'),'pos':self._vec(j,'pos',[0,0,0]),'axis':self._vec(j,'axis',[0,0,1]),'qpos0':float(j.get('ref','0'))}
   idx=len(self.nodes); self.nodes[name]=node; node['idx']=idx
   self._walk(b,name)
 def pose(self,root_pos,root_quat,q):
  poses={}; poses['__root__']=(np.asarray(root_pos,np.float32),qnorm(root_quat))
  pending=True
  while pending:
   pending=False
   for name,n in self.nodes.items():
    if name in poses or n['parent'] not in poses: continue
    pp,pq=poses[n['parent']]; bp=pp+qapply(pq,n['pos']); bq=qmul(pq,n['quat']); j=n['joint']
    if j is not None:
     anchor=bp+qapply(bq,j['pos']); qi=self.jidx[j['name']]; bq=qmul(bq,qaxis(j['axis'],float(q[qi])-j['qpos0'])); bp=anchor-qapply(bq,j['pos'])
    poses[name]=(bp,bq); pending=True
  if 'torso_link' not in poses: raise KeyError('torso_link missing from XML')
  return poses['torso_link']

class State(enum.Enum): DAMPING='damping'; ALIGN='moving to first frame'; READY='holding first frame'; POLICY='policy'
class ReproducedController:
 def __init__(self,assets,policy_path,motion='walk1_subject1',align_seconds=8.0,skip_start_pose_check=False):
  assets=Path(assets); self.cfg=json.loads((assets/'config.json').read_text()); self.q0=np.asarray(self.cfg['q0'],np.float32); self.scale=np.asarray(self.cfg['scale'],np.float32); self.kp=np.asarray(self.cfg['kp'],np.float32); self.kd=np.asarray(self.cfg['kd'],np.float32); self.effort=np.asarray(self.cfg['effort'],np.float32)
  self.joint_names=tuple(self.cfg['joint_names']); self.fk=AnchorFK(assets/'robot/g1.xml',self.joint_names); self.align_seconds=float(align_seconds); self.skip_start_pose_check=bool(skip_start_pose_check)
  self.session=ort.InferenceSession(str(policy_path),providers=['CPUExecutionProvider']); ins=self.session.get_inputs(); self.goal_dim=int(ins[1].shape[-1]);
  if ins[0].shape[-1]!=930 or self.goal_dim!=689: raise ValueError(f'expected reproduced inputs (930,689), got {ins[0].shape}')
  self.motion_dir=assets/('reproduced_motions' if (assets/'reproduced_motions').exists() else 'motions'); self.select_motion(motion); self.state=State.DAMPING; self.frame=0; self.history=np.zeros((10,93),np.float32); self.previous_action=np.zeros(29,np.float32); self.last_keys=0; self.reason='startup'
  self.current_root_pos_live=None; self.current_root_quat_live=None
  self.ref_align_pos=np.zeros(3,np.float32); self.ref_align_quat=np.array([1,0,0,0],np.float32)
 def select_motion(self,motion):
  with np.load(self.motion_dir/f'{motion}.npz') as d:
   req=('goal','joint_pos','joint_vel','root_pos','root_quat','root_ang_vel_b','anchor_pos_w','anchor_quat_w')
   if any(k not in d for k in req): raise ValueError(f'{motion} lacks reproduced anchor fields; run add_anchor_pose.py')
   self.goals=d['goal'].astype(np.float32); self.positions=d['joint_pos'].astype(np.float32); self.root_pos=d['root_pos'].astype(np.float32); self.root_quat=d['root_quat'].astype(np.float32); self.anchor_pos=d['anchor_pos_w'].astype(np.float32); self.anchor_quat=d['anchor_quat_w'].astype(np.float32)
  self.motion=motion; self.motions=tuple(sorted(p.stem for p in self.motion_dir.glob('*.npz')))
 def damp(self,reason): self.state=State.DAMPING; self.reason=reason; self.previous_action.fill(0)
 def obs(self,q,dq,omega,quat): return np.concatenate((q-self.q0,dq,omega,gravity(quat),self.previous_action)).astype(np.float32)
 def goal(self,q,quat):
  if self.current_root_pos_live is None: raise RuntimeError('root pose is required for anchor_obs policy')
  rp,rq=self.fk.pose(self.current_root_pos_live,self.current_root_quat_live,q)
  idx=np.minimum(np.arange(self.frame,self.frame+10),len(self.goals)-1); g=self.goals[idx].reshape(-1)
  refp=self.ref_align_pos+qapply(self.ref_align_quat,self.anchor_pos[self.frame]); refq=qmul(self.ref_align_quat,self.anchor_quat[self.frame])
  relp=qapply(qinv(rq),refp-rp); relq=qmul(qinv(rq),refq)
  return np.concatenate((g,relp,rot6(relq))).astype(np.float32)[None]
 def calibrate_reference_frame(self,q):
  """Yaw-align the motion world frame to the live odometry frame at A.

  The learned goal uses torso-relative reference pose, but the recorded motion
  has an arbitrary world origin/yaw.  A real robot must not inherit that origin.
  Keep the reference gravity unchanged and align only yaw plus translation.
  """
  if self.current_root_pos_live is None: raise RuntimeError('root pose is required for reference alignment')
  live_p,live_q=self.fk.pose(self.current_root_pos_live,self.current_root_quat_live,q)
  ref_p,ref_q=self.anchor_pos[0],self.anchor_quat[0]
  self.ref_align_quat=qmul(qyaw(live_q),qinv(qyaw(ref_q)))
  self.ref_align_pos=live_p-qapply(self.ref_align_quat,ref_p)
 def tick(self,now,q,dq,quat,omega,keys):
  rising=keys & ~self.last_keys; self.last_keys=keys
  if keys&B: self.damp('B pressed'); return None
  if rising&(X|Y) and self.state==State.DAMPING:
   i=self.motions.index(self.motion); self.select_motion(self.motions[(i+(-1 if rising&X else 1))%len(self.motions)]); self.frame=0; self.reason=f'selected {self.motion}'; return None
  if rising&START and self.state in (State.DAMPING,State.READY): self.start_q=q.copy(); self.start_time=now; self.state=State.ALIGN; self.frame=0; self.history.fill(0); self.previous_action.fill(0); self.reason='Aligning to first frame'
  if self.state==State.ALIGN:
   t=np.clip((now-self.start_time)/self.align_seconds,0,1); b=t*t*(3-2*t); target=self.start_q*(1-b)+self.positions[0]*b
   if t>=1: self.state=State.READY; self.reason='Holding first frame; release and press A'
   return target
  if self.state==State.READY:
   if rising&A:
    errors=np.abs(q-self.positions[0]); grav_err=float(np.linalg.norm(gravity(quat)-gravity(self.root_quat[0])))
    if not self.skip_start_pose_check and (float(errors.max())>=0.15 or grav_err>=0.35):
     i=int(np.argmax(errors)); self.reason=f'A rejected: largest joint error {self.joint_names[i]}={errors[i]:.3f} rad (required <0.150), gravity direction error={grav_err:.3f} (required <0.350); release and press A again'; return self.positions[0].copy()
    self.calibrate_reference_frame(q)
    self.state=State.POLICY; self.reason='Policy running'; self.history[:]=self.obs(q,dq,omega,quat)
   else: return self.positions[0].copy()
  if self.state==State.POLICY:
   if self.frame>=len(self.goals): self.damp('motion complete'); return None
   self.history[:-1]=self.history[1:]; self.history[-1]=self.obs(q,dq,omega,quat); out=self.session.run(None,{'obs_history':flatten_history(self.history),'goal':self.goal(q,quat)})[0].reshape(29).astype(np.float32)
   if not np.isfinite(out).all(): self.damp('nonfinite action'); return None
   self.previous_action=out; self.frame+=1; return self.q0+self.scale*out
  return None
 def set_root(self,pos,quat): self.current_root_pos_live=np.asarray(pos,np.float32); self.current_root_quat_live=np.asarray(quat,np.float32)
 def bounded_target(self,target,q,dq):
  tau=np.clip(self.kp*(target-q)-self.kd*dq,-self.effort,self.effort); return q+(tau+self.kd*dq)/self.kp
