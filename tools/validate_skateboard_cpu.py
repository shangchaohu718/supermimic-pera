#!/usr/bin/env python3
"""Offline CPU integration report for the native G1 + HUSKY skateboard.

Runs fixed PD targets for diagnostics; does not load or train a policy.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MUJOCO_GL'] = 'disable'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import mujoco
import numpy as np
from unisupermimic.interaction.skateboard_scene import G1SkateboardCPU, BOARD_XML


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    env=G1SkateboardCPU()
    (args.output/'scene.xml').write_text(env.xml)
    summary={'purpose':'CPU scene/contact integration; no policy training or balance claim',
             'device':'CPU', 'mujoco_version':mujoco.__version__,'download_bytes':0,
             'model':{'nq':env.model.nq,'nv':env.model.nv,'nu':env.model.nu,'policy_actions':29},
             'board_source':json.loads(BOARD_XML.with_suffix('.source.json').read_text()),
             'board_xml_sha256':hashlib.sha256(BOARD_XML.read_bytes()).hexdigest(),
             'scenarios':{}}
    all_checks=[]
    for placement in ('board','ground'):
        env.reset(placement)
        rows=[]
        state_at_load=None
        for i in range(100):
            obs=env.step(np.zeros(29))
            groups=obs['interaction']['contacts']
            rows.append([obs['time_s'],obs['robot_qpos'][2],
                         *[groups[k]['normal_force'] for k in groups]])
            if i==19:
                state_at_load=obs
                (args.output/f'{placement}_load_state.json').write_text(json.dumps(obs,indent=2)+'\n')
                np.savez_compressed(args.output/f'{placement}_load_pose.npz',qpos=env.data.qpos)
        loaded=state_at_load['interaction']['contacts']
        support='board' if placement=='board' else 'ground'
        other='ground' if placement=='board' else 'board'
        checks={
            'both_feet_load_expected_surface':all(loaded[f'{s}_foot_{support}']['normal_force']>50 for s in ('left','right')),
            'no_contact_with_other_surface':all(loaded[f'{s}_foot_{other}']['active_count']==0 for s in ('left','right')),
            'wheel_ground_force_positive':loaded['wheel_ground']['force_world'][2]>50,
            'no_numerical_warnings':not any(w.number for w in env.data.warning),
            'finite_state':bool(np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()),
        }
        summary['scenarios'][placement]={
            'checks':checks,'contact_check_time_s':.4,
            'numerical_check_duration_s':float(env.data.time),
            'simulated_seconds':float(env.data.time),
            'load_contacts_at_0_4s':loaded,
            'minimum_pelvis_height_over_2s':min(row[1] for row in rows),
            'fell_below_0_45m_with_fixed_PD':any(row[1]<.45 for row in rows),
            'note':'Fixed PD has no learned balance; later falling is expected and recorded, not hidden.'}
        all_checks.extend(checks.values())
        with (args.output/f'{placement}_metrics.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['time_s','pelvis_height',*groups.keys()]);writer.writerows(rows)
    summary['all_integration_checks_passed']=all(all_checks)
    summary['training_runtime_imported']=any(name in sys.modules for name in ('torch','warp'))
    assert not summary['training_runtime_imported']
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'checks_passed':all(all_checks),
                      'scenarios':{k:{'fell_with_fixed_PD':v['fell_below_0_45m_with_fixed_PD'],
                                      'wheel_load_N_at_0_4s':v['load_contacts_at_0_4s']['wheel_ground']['force_world'][2]}
                                   for k,v in summary['scenarios'].items()}},indent=2))
    if not all(all_checks):
        raise SystemExit(1)


if __name__=='__main__':
    main()
