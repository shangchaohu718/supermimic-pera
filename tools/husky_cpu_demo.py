#!/usr/bin/env python3
"""Offline HUSKY checkpoint evaluation; CPU physics, inference and rendering.

Observation/control logic adapted from TeleHuman/humanoid_skateboarding,
test_scene/sim.py, commit d93833e80deff7f927c0b80ef9c435d8b5c488fe.
This adapter is subject to that project's CC BY-NC 4.0 license:
https://github.com/TeleHuman/humanoid_skateboarding/blob/main/LICENSE-CC-BY-NC-4.0.md
It evaluates the upstream policy, not a trained SuperMimic-PERA policy.
"""
import argparse
from collections import deque
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

# Set these before importing MuJoCo/OpenGL/ONNX Runtime. No CUDA or GPU rendering.
os.environ.update(CUDA_VISIBLE_DEVICES="", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                  EGL_PLATFORM="surfaceless", LIBGL_ALWAYS_SOFTWARE="1",
                  OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", LP_NUM_THREADS="2")
mesa = Path('/usr/share/glvnd/egl_vendor.d/50_mesa.json')
if mesa.exists():
    os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = str(mesa)

import numpy as np
import mujoco
import onnxruntime as ort

REINDEX = [15, 16, 17, 18, 19, 20, 21, 22, 0, 2, 6, 8, 12, 1, 3, 7, 9, 13, 14, 4, 5, 10, 11]
DEFAULT = np.array([0, 0, 0, .23, -.20, 0, -.7, 0, 0, 1.17, -.45, 0,
                    0, 0, 0, -.03, .45, -.21, 1.32, -.7, -.845, .83, 1.19])
SCALE = np.array([.5475, .3507, .5475, .3507, .4386, .4386,
                  .5475, .3507, .5475, .3507, .4386, .4386,
                  .5475, .4386, .4386] + [.4386] * 8)
DIMS = [2, 1, 3, 3, 23, 23, 23, 1]
STARTS = np.cumsum([0] + DIMS[:-1])


def command(t, scenario):
    # h is absolute heading in radians, not yaw rate.
    if scenario == 'turn':
        return 1.0, (.35 if 6 <= t < 18 else 0.0)
    return 1.0, 0.0


def run(args):
    root = args.husky_root.resolve()
    policy = root / 'ckpts/test.onnx'
    xml = root / 'test_scene/mjlab_scene.xml'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(policy), sess_options=opts, providers=['CPUExecutionProvider'])
    inp = session.get_inputs()[0]
    assert inp.shape == [1, 395], inp.shape
    model = mujoco.MjModel.from_xml_path(str(xml))
    assert (model.nq, model.nv, model.nu) == (44, 42, 30)
    model.opt.timestep = .005
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    model.opt.ccd_iterations = 50
    data = mujoco.MjData(model)
    data.qpos[:] = np.concatenate([[-.03, .1, .78], [1, 0, 0, 0], DEFAULT,
                                   [0, 0, .1], [1, 0, 0, 0], np.zeros(7)])
    data.ctrl[:-7] = DEFAULT[REINDEX]
    mujoco.mj_forward(model, data)
    history = deque([np.zeros(79, np.float32) for _ in range(5)], maxlen=5)
    action = np.zeros(23, np.float32)
    sensor = model.sensor('robot/imu_ang_vel').id
    adr = model.sensor_adr[sensor]
    pelvis = model.body('robot/pelvis').id
    board = model.body('skateboard/skateboard_deck').id
    rows, states = [], []
    started = time.monotonic()
    for step in range(round(args.seconds * 50)):
        v, h = command(step * .02, args.scenario)
        w, x, y, z = data.qpos[3:7]
        gravity = [2*(-z*x+w*y), -2*(z*y+w*x), 1-2*(w*w+z*z)]
        heading = np.arctan2(2*(x*y+w*z), 1-2*(y*y+z*z))
        phase = ((step + 1) * .02 / 6) % 1
        obs = np.concatenate([[2*v, h, heading/np.pi], data.sensordata[adr:adr+3]*.25,
                              gravity, data.qpos[7:30]-DEFAULT, data.qvel[6:29]*.05,
                              action, [phase]])
        history.append(obs)
        hist = np.asarray(history)
        stacked = np.concatenate([hist[:, start:start+dim].ravel()
                                  for start, dim in zip(STARTS, DIMS)]).astype(np.float32)
        action = session.run(None, {inp.name: stacked[None]})[0].reshape(23)
        if not np.isfinite(action).all():
            raise RuntimeError('Non-finite policy output')
        data.ctrl[:-7] = (DEFAULT + action*SCALE)[REINDEX]
        for _ in range(4):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError('Non-finite simulator state')
        rows.append([data.time, v, h, *data.xpos[pelvis], *data.xpos[board], heading,
                     float(np.linalg.norm(data.xpos[pelvis, :2]-data.xpos[board, :2]))])
        if step % 2 == 0:
            states.append(data.qpos.copy())
    rows = np.asarray(rows)
    np.savez_compressed(out/'trajectory.npz', qpos=np.asarray(states), fps=25)
    with (out/'metrics.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s','command_speed','command_heading','pelvis_x','pelvis_y',
                         'pelvis_z','board_x','board_y','board_z','heading_before_step',
                         'pelvis_board_xy_distance'])
        writer.writerows(rows)
    report = {
        'policy_source': 'HUSKY provided test.onnx; no new training',
        'supermimic_pera_policy': False,
        'husky_commit': subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True).strip(),
        'policy_sha256': hashlib.sha256(policy.read_bytes()).hexdigest(),
        'scene_sha256': hashlib.sha256(xml.read_bytes()).hexdigest(),
        'scenario': args.scenario, 'simulated_seconds': float(data.time),
        'physics_hz': 200, 'policy_hz': 50, 'video_fps': 25,
        'inference_providers': session.get_providers(), 'physics': 'MuJoCo CPU',
        'mujoco_version': mujoco.__version__, 'onnxruntime_version': ort.__version__,
        'new_download_bytes': 0, 'rollout_wall_seconds': time.monotonic()-started,
        'pelvis_min_height_m': float(rows[:,5].min()),
        'pelvis_final_height_m': float(rows[-1,5]),
        'board_net_displacement_m': float(np.linalg.norm(rows[-1,6:8]-rows[0,6:8])),
        'board_path_length_m': float(np.linalg.norm(np.diff(rows[:,6:8],axis=0),axis=1).sum()),
        'pelvis_board_max_xy_distance_m': float(rows[:,-1].max()),
        'height_below_0_45m': bool((rows[:,5]<.45).any()),
        'simulation_warning_counts': [int(w.number) for w in data.warning],
        'note': 'Single deterministic rollout; height threshold is only a fall diagnostic, not a success benchmark.'
    }
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    if args.render:
        render(model, states, out, args, report)


def render(model, states, out, args, report):
    from OpenGL import GL
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 640
    model.vis.quality.shadowsize = 1024
    renderer = mujoco.Renderer(model, 640, 960)
    gl_renderer = GL.glGetString(GL.GL_RENDERER).decode()
    if not any(s in gl_renderer.lower() for s in ('llvmpipe', 'softpipe', 'software')):
        renderer.close()
        raise RuntimeError(f'Expected CPU renderer, got {gl_renderer}')
    report['renderer'] = gl_renderer
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.5, 135, -18
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    option.sitegroup[:] = 0
    font = ImageFont.truetype('DejaVuSans.ttf',18)
    started = time.monotonic()
    with imageio.get_writer(out/'husky_cpu_demo.mp4',fps=25,codec='libx264',quality=7,
                            macro_block_size=16,ffmpeg_params=['-threads','2']) as writer:
        for i, state in enumerate(states):
            data.qpos[:] = state
            mujoco.mj_forward(model,data)
            cam.lookat[:] = data.xpos[model.body('robot/pelvis').id]
            cam.lookat[2] = .5
            renderer.update_scene(data,camera=cam,scene_option=option)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0,0,960,58),fill=(18,24,32))
            draw.text((16,7),'HUSKY pretrained policy | CPU physics + inference + rendering',font=font,fill='white')
            v,h = command(i/25,args.scenario)
            draw.text((16,32),f'{args.scenario} | t={i/25:.2f}s | target {v:.1f} m/s, heading {h:.2f} rad | not PERA trained',font=font,fill=(175,220,240))
            writer.append_data(np.asarray(frame))
            if i in (0,len(states)//2,len(states)-1):
                frame.save(out/f'frame_{i:04d}.png')
            if i % 250 == 0:
                print(f'Rendered {i}/{len(states)} frames ({time.monotonic()-started:.1f}s)',flush=True)
    renderer.close()
    print(f'Video: {out / "husky_cpu_demo.mp4"}',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--husky-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--scenario',choices=['straight','turn'],default='straight')
    parser.add_argument('--render',action='store_true')
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    run(args)
