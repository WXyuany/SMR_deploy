#!/usr/bin/env python3
"""Export a SuperMimic-Reproduced PyTorch checkpoint's policy only.

The resulting ONNX has obs_history(1,930), goal(1,680 or 689), action(1,29).
"""
import argparse
from pathlib import Path
import torch
from supermimic.policy.nets import Policy, PolicyCfg

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); pc=ck['config']['policy']; cfg=PolicyCfg(**{k:(tuple(v) if isinstance(v,list) else v) for k,v in pc.items()}); model=Policy(cfg).cpu().eval(); model.load_state_dict(ck['policy']); model.eval(); obs=torch.zeros(1,930); goal=torch.zeros(1,cfg.goal_dim); a.output.parent.mkdir(parents=True,exist_ok=True); torch.onnx.export(model,(obs,goal),a.output,input_names=['obs_history','goal'],output_names=['action'],dynamic_axes=None,opset_version=17,dynamo=False,do_constant_folding=True); print({'iteration':ck['iteration'],'goal_dim':cfg.goal_dim,'action_dim':29,'output':str(a.output)})
if __name__=='__main__': main()
