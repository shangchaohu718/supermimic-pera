#!/usr/bin/env python3
"""Offline mixed-task SuperMimic-PERA prototype: CPU physics + small GPU learner.

One frozen native foundation, one shared interaction residual, joint robot/object
world model, and critic. This is an experimental model-based training entry,
not the unchanged upstream benchmark or a claim of learned task completion.
"""
import argparse
from collections import deque
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

os.environ.update(MUJOCO_GL='disable',UV_OFFLINE='1',HF_HUB_OFFLINE='1',WANDB_MODE='offline',
                  OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from unisupermimic.interaction.multitask_scene import G1InteractionCPU,TASKS
from unisupermimic.interaction.networks import RetainedInteractionPolicy,replay_action_loss
from unisupermimic.interaction.world_model import JointInteractionWorldModel
from unisupermimic.interaction.references import ReferenceClip
from unisupermimic.interaction.features import state_from_observation,native_proprio,context_features,reward_and_done
from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.networks import WorldModel,Critic,run_mlp
from unisupermimic.rl.algos.supermimic.geometry import flatten_state,unflatten_state


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


class BalancedReplay:
    def __init__(self,capacity=4096):
        self.tasks=[deque(maxlen=capacity) for _ in TASKS]
    def add(self,task,item):
        self.tasks[task].append(item)
    def sample(self,per_task=4):
        if any(not b for b in self.tasks):
            raise ValueError('Every task must have real experience before updating')
        return [random.choice(b) for b in self.tasks for _ in range(per_task)]


class Training:
    def __init__(self,args):
        self.args=args;self.out=args.output;self.out.mkdir(parents=True,exist_ok=False)
        self.device=torch.device('cuda:0' if args.device=='cuda' else args.device);torch.set_num_threads(2)
        networks.AMP_DTYPE=torch.bfloat16 if self.device.type=='cuda' else None
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        if self.device.type=='cuda':
            # Leave room below the 4 GiB process budget for CUDA context/libraries.
            total=torch.cuda.get_device_properties(self.device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(.9,(args.vram_mib-768)*1024**2/total),self.device)
        self.envs=[G1InteractionCPU(task,seed=args.seed) for task in TASKS]
        self.terrain={i:torch.tensor(e.terrain_boxes,dtype=torch.float32,device=self.device) for i,e in enumerate(self.envs)}
        self.default=torch.tensor(self.envs[0].default_q,dtype=torch.float32,device=self.device)
        self.clips=[ReferenceClip.from_csv(p,self.envs[0].joint_names) for p in sorted(args.motions.glob('*.csv'))]
        if not self.clips:raise ValueError('Local motion CSV files required')
        self.tracking_count=len(self.clips)
        walk=next((i for i,c in enumerate(self.clips) if c.name.startswith('walk')),0)
        self.task_clips=[0,walk]
        # Same flat carrying prior in both box scenes. The local box_stair
        # trajectory belongs to a different staircase geometry and is not used.
        for directory in ('box_1','box_1'):
            self.task_clips.append(len(self.clips))
            self.clips.append(ReferenceClip.from_vaic(args.vaic_root/directory,self.envs[0].joint_names))
        # Board starts with native 29-DoF stance; HUSKY/VAIC policies are not imported.
        initial=self.envs[4].observe();q=torch.tensor(initial['robot_qpos'],dtype=torch.float32)
        self.task_clips.append(len(self.clips))
        self.clips.append(ReferenceClip('board_stance',self.envs[0].joint_names,
                           q[:3].repeat(1000,1),q[3:7].repeat(1000,1),q[7:].repeat(1000,1)))
        print(f'Loaded {self.tracking_count} native motions + two local box references + board stance',flush=True)
        self.pi=RetainedInteractionPolicy.from_native_checkpoint(args.checkpoint).to(self.device)
        ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        base=WorldModel();base.load_state_dict(ckpt['wm'])
        self.wm=JointInteractionWorldModel(base).to(self.device)
        self.critic=Critic(71+27+64+32,hidden=(128,128)).to(self.device)
        self.opt_pi=torch.optim.Adam(self.pi.interaction_parameters(),lr=1e-4)
        self.opt_wm=torch.optim.Adam(self.wm.correction.parameters(),lr=2e-4)
        self.opt_v=torch.optim.Adam(self.critic.parameters(),lr=2e-4)
        self.replay=BalancedReplay();self.iteration=0;self.stop=False;self.accepted=0;self.rejected=0
        self.started=time.time();self.peak_process_mib=0;self.anchor=None
        self.foundation_sha=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        self.stable_pi=copy.deepcopy(self.pi.residual.state_dict())
        self.frames=[0]*5;self.clip_ids=self.task_clips.copy()
        self.hist=torch.zeros(5,10,99,device=self.device);self.prev=torch.zeros(5,29,device=self.device)
        for i in range(5):self.reset(i)
        self.stable_metrics=self.evaluate()
        self.best_returns={k:v['return'] for k,v in self.stable_metrics.items()}
        self.retention=self.tracking_retention()
        if args.resume:
            saved=torch.load(args.resume,map_location=self.device,weights_only=True)
            if saved.get('schema')!=2 or saved.get('foundation_sha256')!=self.foundation_sha:
                raise ValueError('Resume requires matching schema and native foundation')
            if saved.get('eval_steps')!=args.eval_steps:
                raise ValueError('Resume must preserve the physical evaluation horizon')
            for module,key in ((self.pi,'policy'),(self.wm,'world_model'),(self.critic,'critic')):
                module.load_state_dict(saved[key])
            for opt,key in ((self.opt_pi,'opt_pi'),(self.opt_wm,'opt_wm'),(self.opt_v,'opt_v')):
                opt.load_state_dict(saved[key])
            self.pi.residual.load_state_dict(saved['stable_actor'])
            self.opt_pi.state.clear() # candidate optimizer may differ from accepted actor
            self.stable_pi=copy.deepcopy(saved['stable_actor']);self.best_returns=saved['best_returns']
            self.iteration=saved['iteration'];self.stable_metrics=self.evaluate()
            self.retention=self.tracking_retention()
            del saved
        self.memory_check()
        self.save('accepted.pt')
        atomic_json(self.out/'provenance.json',{
            'native_checkpoint':str(args.checkpoint),'native_iteration':ckpt['iteration'],
            'checkpoint_sha256':self.foundation_sha,
            'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'tasks':list(TASKS),'local_reference_names':[c.name for c in self.clips],
            'shared_actor':True,'native_foundation_frozen':True,'new_download_bytes':0,
            'algorithm':'Native Actor/WM foundation + shared interaction residual; real-transition WM supervision and differentiable imagined actor/critic updates; no PPO',
            'prototype_limits':'One free-object slot (box or articulated board); box+board composition not yet supported; short-horizon initial curriculum; no task success claim',
            'baseline_task_returns':self.stable_metrics,'native_retention':self.retention})
        del ckpt

    def tensor(self,value):return torch.as_tensor(value,dtype=torch.float32,device=self.device)

    def value_input(self,state,obj,ctx,goal):
        latent=torch.tanh(run_mlp(self.pi.base.enc,self.pi.base.goal_norm(goal))).detach()
        return torch.cat((flatten_state(state),obj,ctx,latent),-1)

    def reference(self,clip_ids,frames):
        indexes=[min(int(f),self.clips[int(c)].num_frames-1) for c,f in zip(clip_ids,frames)]
        state=self.tensor(torch.stack([self.clips[int(c)].state[f] for c,f in zip(clip_ids,indexes)]).numpy())
        goal=torch.cat([self.clips[int(c)].goal(torch.tensor([f])) for c,f in zip(clip_ids,indexes)]).to(self.device)
        obj=self.tensor(np.stack([self.clips[int(c)].objects[f] for c,f in zip(clip_ids,indexes)]))
        return unflatten_state(state),goal,obj

    def targets(self,reference,reference_obj,task_ids):
        return torch.where(((task_ids==2)|(task_ids==3))[:,None],reference_obj[:,:3],reference[0])

    def context(self,state,obj,task_ids,reference,reference_obj):
        targets=self.targets(reference,reference_obj,task_ids)
        heights=torch.stack((reference[0][:,2],targets[:,2]),-1)
        return context_features(state,obj,task_ids,targets,heights,self.terrain)

    def reset(self,i):
        if i==0:self.clip_ids[i]=random.randrange(self.tracking_count)
        c=self.clips[self.clip_ids[i]];self.frames[i]=0
        object_state=c.objects[0] if i in (2,3) else None
        if object_state is not None:
            object_state=object_state.copy();object_state[2]=max(.181,object_state[2])
        self.envs[i].reset(c.qpos[0],c.qvel[0],object_state)
        self.hist[i].zero_();self.prev[i].zero_()

    @torch.no_grad()
    def collect(self):
        ids=torch.arange(5,device=self.device);mask=(ids!=0).float()
        for _ in range(self.args.collect_steps):
            observations=[e.observe() for e in self.envs]
            state=tuple(torch.cat([state_from_observation(o,self.device)[j] for o in observations]) for j in range(6))
            obj=self.tensor(np.stack([o['object_state'] for o in observations]))
            ref,goal,ref_obj=self.reference(self.clip_ids,self.frames)
            current=native_proprio(state,self.prev,ref,self.default)
            self.hist=torch.cat((self.hist[:,1:],current[:,None]),1)
            for i,f in enumerate(self.frames):
                if f==0:self.hist[i]=current[i][None].expand(10,-1)
            context=self.context(state,obj,ids,ref,ref_obj)
            actions=self.pi(self.hist,goal,context,mask)
            actions=actions+torch.randn_like(actions)*.03*mask[:,None]
            ns=[];no=[]
            for i,e in enumerate(self.envs):
                obs=e.step(actions[i].cpu().numpy());ns.append(flatten_state(state_from_observation(obs,'cpu'))[0].numpy());no.append(obs['object_state'])
            flat=flatten_state(state).cpu().numpy()
            for i in range(5):
                self.replay.add(i,{'s':flat[i].copy(),'o':obj[i].cpu().numpy().copy(),
                    'ns':ns[i],'no':np.asarray(no[i],np.float32),'a':actions[i].cpu().numpy().copy(),
                    'h':self.hist[i].cpu().numpy().copy(),'p':self.prev[i].cpu().numpy().copy(),
                    'c':self.clip_ids[i],'f':self.frames[i],'t':i})
                self.frames[i]+=1;self.prev[i]=actions[i]
                # Curriculum episodes, not complete reference evaluations.
                if self.frames[i]>=min(self.args.episode_steps,self.clips[self.clip_ids[i]].num_frames-11) or ns[i][2]<.4:
                    self.reset(i)

    def batch(self):
        rows=self.replay.sample(self.args.per_task_batch)
        b={k:self.tensor(np.stack([r[k] for r in rows])) for k in ('s','o','ns','no','a','h','p')}
        b.update({k:[r[k] for r in rows] for k in ('c','f')})
        b['t']=torch.tensor([r['t'] for r in rows],device=self.device)
        return b

    def update(self):
        b=self.batch();s=unflatten_state(b['s']);ns=unflatten_state(b['ns']);mask=(b['t']!=0).float()
        ref,goal,ref_obj=self.reference(b['c'],b['f']);ctx=self.context(s,b['o'],b['t'],ref,ref_obj)
        wm_loss=self.wm.loss(s,b['o'],b['a'],ctx,mask,ns,b['no'],update_norm=True)
        if not torch.isfinite(wm_loss):raise RuntimeError('Non-finite world-model loss')
        self.opt_wm.zero_grad();wm_loss.backward();torch.nn.utils.clip_grad_norm_(self.wm.correction.parameters(),1);self.opt_wm.step()
        next_ref,next_goal,next_obj_ref=self.reference(b['c'],[f+1 for f in b['f']])
        nctx=self.context(ns,b['no'],b['t'],next_ref,next_obj_ref)
        vin=self.value_input(s,b['o'],ctx,goal)
        self.critic.norm.update(vin.detach())
        with torch.no_grad():
            reward,done=reward_and_done(ns,b['no'],b['a'],b['p'],next_ref,self.targets(next_ref,next_obj_ref,b['t']),b['t'],self.terrain)
            target=reward+.99*(1-done)*self.critic(self.value_input(ns,b['no'],nctx,next_goal))
        v_loss=(self.critic(vin)-target).square().mean()
        if not torch.isfinite(v_loss):raise RuntimeError('Non-finite critic loss')
        self.opt_v.zero_grad();v_loss.backward();torch.nn.utils.clip_grad_norm_(self.critic.parameters(),1);self.opt_v.step()
        pi_loss=torch.zeros((),device=self.device)
        if self.iteration>=self.args.warmup and min(map(len,self.replay.tasks))>=8:
            self.wm.requires_grad_(False);self.critic.requires_grad_(False)
            hist=b['h'];obj=b['o'];previous=b['p'];value=0;alive=torch.ones_like(mask)
            try:
                for step in range(self.args.horizon):
                    ref,goal,ref_obj=self.reference(b['c'],[f+step for f in b['f']])
                    ctx=self.context(s,obj,b['t'],ref,ref_obj) # re-query terrain at predicted state
                    action=self.pi(hist,goal,ctx,mask)
                    s,obj=self.wm.step(s,obj,action,ctx,mask)
                    next_ref,next_goal,next_ref_obj=self.reference(b['c'],[f+step+1 for f in b['f']])
                    r,d=reward_and_done(s,obj,action,previous,next_ref,self.targets(next_ref,next_ref_obj,b['t']),b['t'],self.terrain)
                    value=value+(.99**step)*alive*r;alive=alive*(1-d)
                    hist=torch.cat((hist[:,1:],native_proprio(s,action,next_ref,self.default)[:,None]),1);previous=action
                ctx=self.context(s,obj,b['t'],next_ref,next_ref_obj)
                value=value+(.99**self.args.horizon)*alive*self.critic(self.value_input(s,obj,ctx,next_goal))
                pi_loss=-(value*mask).sum()/mask.sum()
                if self.anchor is not None:
                    h,g,c,m,a=self.anchor
                    pi_loss=pi_loss+.5*replay_action_loss(self.pi(h,g,c,m),a)
                if not torch.isfinite(pi_loss):raise RuntimeError('Non-finite actor loss')
                self.opt_pi.zero_grad();pi_loss.backward();torch.nn.utils.clip_grad_norm_(self.pi.interaction_parameters(),.5);self.opt_pi.step()
            finally:
                self.wm.correction.requires_grad_(True);self.critic.requires_grad_(True)
        return {'wm_loss':float(wm_loss.detach()),'critic_loss':float(v_loss.detach()),'actor_loss':float(pi_loss.detach())}

    @torch.no_grad()
    def tracking_retention(self):
        errors=[]
        for c in range(self.tracking_count):
            ref,goal,obj=self.reference([c],[0]);h=native_proprio(ref,torch.zeros(1,29,device=self.device),ref,self.default)[:,None].repeat(1,10,1)
            native=self.pi.base(h,goal);new=self.pi(h,goal,torch.zeros(1,64,device=self.device),torch.zeros(1,device=self.device))
            errors.append(float((native-new).abs().max()))
        if any(errors):raise RuntimeError('Native tracking actor regression')
        return {'clips_checked':len(errors),'max_action_error':max(errors),'scope':'First reference state per native clip; frozen function and original tracking scene preserved; not full-clip success evaluation'}

    @torch.no_grad()
    def evaluate(self):
        result={}
        for task in range(1,5):
            env=G1InteractionCPU(TASKS[task],seed=123)
            c=self.clips[self.task_clips[task]]
            initial_obj=c.objects[0].copy() if task in (2,3) else None
            if initial_obj is not None:initial_obj[2]=max(.181,initial_obj[2])
            obs=env.reset(c.qpos[0],c.qvel[0],initial_obj)
            h=torch.zeros(1,10,99,device=self.device);prev=torch.zeros(1,29,device=self.device)
            tid=torch.tensor([task],device=self.device);total=0;step=0
            for step in range(self.args.eval_steps):
                s=state_from_observation(obs,self.device);obj=self.tensor(obs['object_state'])[None]
                ref,goal,ro=self.reference([self.task_clips[task]],[step])
                current=native_proprio(s,prev,ref,self.default)
                h=(current[:,None].expand(-1,10,-1).clone() if step==0
                   else torch.cat((h[:,1:],current[:,None]),1))
                a=self.pi(h,goal,self.context(s,obj,tid,ref,ro),torch.ones(1,device=self.device))
                obs=env.step(a[0].cpu().numpy());ns=state_from_observation(obs,self.device);no=self.tensor(obs['object_state'])[None]
                nr,_,nro=self.reference([self.task_clips[task]],[step+1])
                r,_=reward_and_done(ns,no,a,prev,nr,self.targets(nr,nro,tid),tid,self.terrain)
                total+=float(r);prev=a
                if obs['robot_qpos'][2]<.4:break
            result[TASKS[task]]={'return':total,'steps':step+1,'pelvis_height':float(obs['robot_qpos'][2]),
                                 'scope':'fixed short CPU physics rollout; dense reward, not task success'}
        return result

    def gate(self):
        metrics=self.evaluate();self.retention=self.tracking_retention()
        self.memory_check()
        passed=all(v['return']>=self.best_returns[k]-.02*max(abs(self.best_returns[k]),1)
                   for k,v in metrics.items())
        if passed:
            self.accepted+=1;self.stable_metrics=metrics
            self.best_returns={k:max(self.best_returns[k],v['return']) for k,v in metrics.items()}
            self.stable_pi=copy.deepcopy(self.pi.residual.state_dict());self.save('accepted.pt')
        else:
            self.rejected+=1;self.pi.residual.load_state_dict(self.stable_pi);self.opt_pi.state.clear()
        with torch.no_grad():
            b=self.batch();r,g,o=self.reference(b['c'],b['f']);m=(b['t']!=0).float()
            ctx=self.context(unflatten_state(b['s']),b['o'],b['t'],r,o)
            self.anchor=(b['h'],g,ctx,m,self.pi(b['h'],g,ctx,m).detach())
        atomic_json(self.out/f'eval_{self.iteration:06d}.json',{'accepted':passed,'metrics':metrics,
                    'retention':self.retention,'best_returns':self.best_returns})
        return passed

    def memory_check(self):
        if self.device.type!='cuda':return 0
        output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
        used=next((int(line.split(',')[1]) for line in output.splitlines() if line.split(',')[0].strip()==str(os.getpid())),0)
        self.peak_process_mib=max(self.peak_process_mib,used)
        if used>self.args.vram_mib:raise RuntimeError(f'VRAM budget exceeded: {used} MiB')
        return used

    def save(self,name):
        value={'schema':2,'foundation_sha256':self.foundation_sha,'eval_steps':self.args.eval_steps,
               'iteration':self.iteration,'policy':self.pi.state_dict(),'world_model':self.wm.state_dict(),
               'critic':self.critic.state_dict(),'opt_pi':self.opt_pi.state_dict(),'opt_wm':self.opt_wm.state_dict(),
               'opt_v':self.opt_v.state_dict(),'stable_actor':self.stable_pi,'best_returns':self.best_returns,
               'retention':self.retention,'scope':'experimental mixed interaction; replay/RNG not restored'}
        tmp=self.out/(name+'.tmp');torch.save(value,tmp);tmp.replace(self.out/name)

    def train(self):
        signal.signal(signal.SIGTERM,lambda *_:setattr(self,'stop',True))
        signal.signal(signal.SIGINT,lambda *_:setattr(self,'stop',True))
        try:
            if self.iteration>=self.args.iterations:raise ValueError('iterations must exceed resumed iteration')
            for self.iteration in range(self.iteration+1,self.args.iterations+1):
                self.collect();stats=self.update();used=self.memory_check()
                if self.iteration%self.args.eval_every==0:self.gate();self.save('latest.pt')
                status={'state':'training','pid':os.getpid(),'iteration':self.iteration,
                        'elapsed_seconds':time.time()-self.started,'process_gpu_mib':used,
                        'peak_process_gpu_mib':self.peak_process_mib,
                        'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2 if self.device.type=='cuda' else 0,
                        'replay_by_task':{k:len(b) for k,b in zip(TASKS,self.replay.tasks)},
                        'accepted':self.accepted,'rejected':self.rejected,**stats}
                atomic_json(self.out/'status.json',status)
                if self.iteration%10==0:print(json.dumps(status),flush=True)
                if self.stop:break
                time.sleep(self.args.sleep)
            # Every deliverable actor must pass the physical gate; rejected candidates roll back.
            if self.iteration%self.args.eval_every:
                self.gate()
            self.save('latest.pt')
            status.update(state='paused' if self.stop else 'completed',accepted=self.accepted,
                          rejected=self.rejected,peak_process_gpu_mib=self.peak_process_mib,
                          elapsed_seconds=time.time()-self.started)
            atomic_json(self.out/'status.json',status)
        except Exception as exc:
            atomic_json(self.out/'status.json',{'state':'failed','iteration':self.iteration,'error':repr(exc),'pid':os.getpid()})
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--motions',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--vaic-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--vram-mib',type=int,default=4096);p.add_argument('--warmup',type=int,default=100)
    p.add_argument('--collect-steps',type=int,default=4);p.add_argument('--per-task-batch',type=int,default=4)
    p.add_argument('--horizon',type=int,default=4);p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--eval-steps',type=int,default=256);p.add_argument('--episode-steps',type=int,default=512)
    p.add_argument('--sleep',type=float,default=.1);p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if min(args.iterations,args.collect_steps,args.per_task_batch,args.horizon,args.eval_every,args.eval_steps,args.episode_steps)<=0 or args.vram_mib<1024:
        p.error('Positive sizes and at least 1024 MiB memory budget required')
    Training(args).train()


if __name__=='__main__':main()
