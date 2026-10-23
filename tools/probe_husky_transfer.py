#!/usr/bin/env python3
"""CPU-only feasibility probe: local HUSKY expert on native G1 29-DoF physics.

HUSKY observation mapping is adapted from TeleHuman/humanoid_skateboarding
(test_scene/sim.py, CC-BY-NC-4.0). This is zero-shot evaluation, not training.
No downloads and no mutation of the production robot or contact model.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import argparse
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
import mujoco
import onnxruntime as ort
from unisupermimic.interaction.multitask_scene import G1InteractionCPU
from husky_cpu_demo import DEFAULT,SCALE,DIMS,STARTS

def native_env(husky_root=None, contact_model="native_spheres"):
    """Production contact variant; ``husky_root`` is kept for probe compatibility."""
    return G1InteractionCPU("skateboard", skateboard_foot_contacts=contact_model)

class HuskyNativeExpert:
    def __init__(self, env, husky_root):
        self.env=env
        src=mujoco.MjModel.from_xml_path(str(husky_root/'test_scene/mjlab_scene.xml'))
        names=[src.joint(i).name.removeprefix('robot/') for i in range(src.njnt) if 7<=src.jnt_qposadr[i]<30]
        self.native_indices=np.array([env.joint_names.index(n) for n in names])
        opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(husky_root/'ckpts/test.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
        self.input_name=self.session.get_inputs()[0].name
        assert self.session.get_inputs()[0].shape==[1,395]
        self.reset_history()
    def reset_history(self):
        self.history=deque([np.zeros(79,np.float32) for _ in range(5)],maxlen=5)
        self.action=np.zeros(23,np.float32);self.step_number=0
    def initial_qpos(self, clearance=False):
        e=self.env;q=e.default_qpos.copy();q[:7]=[-.03,.1,.78,1,0,0,0]
        q[7:]=0;q[7+self.native_indices]=DEFAULT
        e.reset(q,np.zeros(35))
        if clearance:
            required=[]
            for side,geoms in e.foot_geoms.items():
                for g in geoms:
                    xyz=e.data.geom_xpos[g];radius=e.model.geom_size[g,0]
                    vertical_extent=radius
                    if e.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_CAPSULE:
                        vertical_extent+=e.model.geom_size[g,1]*abs(e.data.geom_xmat[g,8])
                    support=.115 if abs(xyz[0])<=.4+radius and abs(xyz[1])<=.1+radius else 0
                    required.append(support+.001-(xyz[2]-vertical_extent))
            q[2]+=max(0,max(required));e.reset(q,np.zeros(35))
        return q
    def action_native(self,speed=1.,heading_target=0.):
        e=self.env;d=e.data;w,x,y,z=d.qpos[3:7]
        gravity=[2*(-z*x+w*y),-2*(z*y+w*x),1-2*(w*w+z*z)]
        heading=np.arctan2(2*(x*y+w*z),1-2*(y*y+z*z))
        self.step_number+=1
        phase=(self.step_number*.02/6)%1
        obs=np.r_[2*speed,heading_target,heading/np.pi,d.qvel[3:6]*.25,gravity,
                  d.qpos[7+self.native_indices]-DEFAULT,d.qvel[6+self.native_indices]*.05,self.action,phase]
        self.history.append(obs)
        hist=np.asarray(self.history)
        stacked=np.concatenate([hist[:,s:s+dim].ravel() for s,dim in zip(STARTS,DIMS)]).astype(np.float32)
        self.action=self.session.run(None,{self.input_name:stacked[None]})[0].reshape(23)
        target=np.zeros(29);target[self.native_indices]=DEFAULT+SCALE*self.action
        return (target-e.default_q)/e.action_scale, stacked

def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for clear in ([False,True] if args.placement=='both' else [args.placement=='clearance']):
        env=native_env(args.husky_root,args.contact_model);expert=HuskyNativeExpert(env,args.husky_root)
        q=expert.initial_qpos(clear);expert.reset_history();initial=env.data.qpos.copy()
        states=[];velocities=[];actions=[];observations=[];rows=[]
        start=time.monotonic()
        for step in range(round(args.seconds*50)):
            act,obs=expert.action_native(args.speed)
            states.append(env.data.qpos.copy());velocities.append(env.data.qvel.copy());actions.append(act.copy());observations.append(obs)
            result=env.step(act)
            contacts=result['interaction']['contacts']
            root=np.array(result['robot_qpos'][:3]);board=np.array(result['object_state'][:3])
            rows.append([env.data.time,*root,*board,*[contacts[k]['normal_force'] for k in ('left_foot_board','right_foot_board','left_foot_ground','right_foot_ground','wheel_ground')]])
        rows=np.array(rows);label='clearance' if clear else 'original'
        summary={'placement':label,'contact_model':args.contact_model,'initial_qpos':q.tolist(),'seconds':float(env.data.time),'root_min_height':float(rows[:,3].min()),
                 'first_height_below_045_s':next((float(r[0]) for r in rows if r[3]<.45),None),
                 'board_displacement_m':float(np.linalg.norm(rows[-1,4:6]-initial[36:38])),
                 'foot_board_contact_fraction':float(np.mean(rows[:,7:9].sum(axis=1)>10)),
                 'root_board_max_distance':float(np.linalg.norm(rows[:,1:3]-rows[:,4:6],axis=1).max()),
                 'native_model_nq_nv_nu':[env.model.nq,env.model.nv,env.model.nu],
                 'warnings':[int(w.number) for w in env.data.warning],'wall_seconds':time.monotonic()-start,
                 'device':'CPU','new_download_bytes':0,'training':False}
        np.savez_compressed(args.output/f'{label}_trajectory.npz',qpos=states,qvel=velocities,action=actions,husky_obs=observations,metrics=rows)
        print(json.dumps(summary),flush=True);results.append(summary)
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--husky-root',type=Path,default=Path('/home/tcl/Desktop/start/humanoid_skateboarding'))
    p.add_argument('--output',type=Path,default=Path('outputs/probe_husky_native29'))
    p.add_argument('--seconds',type=float,default=30.)
    p.add_argument('--speed',type=float,default=1.)
    p.add_argument('--contact-model',choices=['native_spheres','husky_capsules'],default='native_spheres')
    p.add_argument('--placement',choices=['original','clearance','both'],default='both')
    run(p.parse_args())
