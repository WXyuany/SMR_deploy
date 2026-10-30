"""Real G1 runner for the 689-dim reproduced policy with LiDAR odometry.

The odometry bridge listens on UDP for newline-delimited JSON.  Example:
  {"position":[x,y,z],"quaternion":[w,x,y,z],"frame":"lidar"}
Use frame=pelvis when the estimator already publishes the pelvis pose.
"""
from __future__ import annotations
import argparse, os, select, signal, struct, sys, termios, threading, time, tty
from pathlib import Path
import numpy as np
from lidar_odometry import UdpOdometry
from reproduced_controller import A, B, START, X, Y, ReproducedController


class Keyboard:
    def __init__(self):
        if not sys.stdin.isatty(): raise RuntimeError("keyboard control requires a terminal")
        self.fd=sys.stdin.fileno(); self.saved=termios.tcgetattr(self.fd); tty.setcbreak(self.fd)
        self.lock=threading.Lock(); self.pulses={}; self.stop=threading.Event()
        self.thread=threading.Thread(target=self._read,daemon=True); self.thread.start()
    def _read(self):
        mp={"s":START,"a":A,"b":B,"x":X,"y":Y}
        try:
            while not self.stop.is_set():
                r,_,_=select.select([self.fd],[],[],.05)
                if not r: continue
                c=os.read(self.fd,1).decode("utf8","ignore").lower()
                if c=="\x03": raise KeyboardInterrupt
                bit=mp.get(c)
                if bit is not None:
                    with self.lock: self.pulses[bit]=time.monotonic()+(.5 if bit==B else .15)
        except KeyboardInterrupt:
            os.kill(os.getpid(),signal.SIGINT)
    def keys(self):
        now=time.monotonic()
        with self.lock:
            self.pulses={b:t for b,t in self.pulses.items() if t>now}
            v=0
            for b in self.pulses: v|=b
            return v
    def close(self):
        self.stop.set(); termios.tcsetattr(self.fd,termios.TCSANOW,self.saved)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--interface',required=True); p.add_argument('--robot',choices=['g1-29dof'],required=True)
    p.add_argument('--assets',type=Path,default=Path(__file__).parent/'assets')
    p.add_argument('--policy',type=Path,default=Path(__file__).parent/'assets/reproduced_policy_lafan1_anchor_5080.onnx')
    p.add_argument('--motion',default='walk1_subject1'); p.add_argument('--align-seconds',type=float,default=8.)
    p.add_argument('--odom-bind',default='0.0.0.0'); p.add_argument('--odom-port',type=int,default=8765)
    p.add_argument('--odom-max-age',type=float,default=.15)
    p.add_argument('--lidar-to-pelvis-pos',type=float,nargs=3,default=(0.,0.,0.))
    p.add_argument('--lidar-to-pelvis-quat',type=float,nargs=4,default=(1.,0.,0.,0.))
    p.add_argument('--skip-start-pose-check',action='store_true'); args=p.parse_args()
    policy=ReproducedController(args.assets,args.policy,args.motion,args.align_seconds,args.skip_start_pose_check)
    odom=UdpOdometry(args.odom_bind,args.odom_port,args.odom_max_age,args.lidar_to_pelvis_pos,args.lidar_to_pelvis_quat)
    keyboard=Keyboard()
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelPublisher,ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_,LowState_
    from unitree_sdk2py.utils.crc import CRC
    ChannelFactoryInitialize(0,args.interface); lock=threading.Lock(); fault=threading.Event(); stop=threading.Event(); shared={'snap':None,'target':None,'stamp':0.,'mode':None}; crc=CRC()
    def receive(msg):
        if CRC().Crc(msg)!=msg.crc:return
        q=np.array([m.q for m in msg.motor_state[:29]],np.float32);dq=np.array([m.dq for m in msg.motor_state[:29]],np.float32);om=np.array(msg.imu_state.gyroscope,np.float32);keys=keyboard.keys()
        with lock:
            if shared['mode'] is not None and shared['mode']!=msg.mode_machine:fault.set()
            shared['snap']=(time.monotonic(),q,dq,om,keys,msg.mode_machine); shared['mode']=shared['mode'] or msg.mode_machine
    sub=ChannelSubscriber('rt/lowstate',LowState_);sub.Init(receive,10)
    deadline=time.monotonic()+10
    while True:
        with lock:snap=shared['snap']
        if snap is not None and odom.latest() is not None:break
        if time.monotonic()>deadline: raise RuntimeError('Need both rt/lowstate and fresh LiDAR odometry before commanding')
        time.sleep(.02)
    switch=MotionSwitcherClient();switch.SetTimeout(5.);switch.Init()
    for _ in range(6):
        code,res=switch.CheckMode()
        if code!=0 or not isinstance(res,dict):raise RuntimeError(f'CheckMode failed: {code}')
        if not res.get('name'):break
        if switch.ReleaseMode()[0]!=0:raise RuntimeError('ReleaseMode failed')
        time.sleep(1.)
    pub=ChannelPublisher('rt/lowcmd',LowCmd_);pub.Init();cmd=unitree_hg_msg_dds__LowCmd_();cmd.mode_pr=0;cmd.mode_machine=shared['mode']
    for i,m in enumerate(cmd.motor_cmd):m.mode=1 if i<29 else 0;m.q=m.dq=m.tau=m.kp=m.kd=0.
    def writer():
        tnext=time.monotonic()
        try:
            while not stop.is_set():
                with lock:snap=shared['snap'];target=shared['target'];stamp=shared['stamp'];mode=shared['mode']
                now=time.monotonic(); stale=now-snap[0]>.1 or snap[-1]!=mode or target is not None and now-stamp>.1; damping=fault.is_set() or target is None or bool(snap[4]&B) or stale
                if not damping:
                    target=policy.bounded_target(target,snap[1],snap[2]);damping=not np.isfinite(target).all()
                for i,m in enumerate(cmd.motor_cmd[:29]):m.q=0. if damping else float(target[i]);m.dq=0.;m.tau=0.;m.kp=0. if damping else float(policy.kp[i]);m.kd=2. if damping else float(policy.kd[i])
                cmd.crc=crc.Crc(cmd);pub.Write(cmd);tnext+=.002;time.sleep(max(0,tnext-time.monotonic()))
        finally: stop.set()
    th=threading.Thread(target=writer,daemon=True);th.start();print('DAMPING. LiDAR odometry ready. s=align, a=run, b=damp, x/y=motion',flush=True)
    last_status=None;tnext=time.monotonic()
    try:
        while not stop.is_set():
            now=time.monotonic()
            with lock:snap=shared['snap']
            pose=odom.latest()
            if pose is None or now-snap[0]>.1 or snap[-1]!=shared['mode']:
                policy.damp('stale LiDAR/LowState watchdog');fault.set();target=None
            elif fault.is_set():
                policy.damp('watchdog');target=None
                if snap[4]&B==0:fault.clear()
            else:
                pos,rq,age=pose;policy.set_root(pos,rq);target=policy.tick(now,snap[1],snap[2],rq,snap[3],snap[4])
            with lock:shared['target']=target;shared['stamp']=time.monotonic()
            status=(policy.state.value,policy.reason,policy.motion)
            if status!=last_status:print(f'{status[0]}: {status[1]}; motion={status[2]}; frame={policy.frame}',flush=True);last_status=status
            tnext+=.02;time.sleep(max(0,tnext-time.monotonic()))
    except KeyboardInterrupt: pass
    finally:
        fault.set();
        with lock:shared['target']=None
        time.sleep(.5);stop.set();th.join(2);keyboard.close();odom.close()

if __name__=='__main__':main()
