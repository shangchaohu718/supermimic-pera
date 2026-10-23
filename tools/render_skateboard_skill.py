#!/usr/bin/env python3
"""Render recorded learned-student physics with Mesa CPU rendering and libx264.

No policy, expert, or training is imported. Video frames replay saved actual
qpos; they do not rerun physics or invent a successful trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

os.environ.update(CUDA_VISIBLE_DEVICES="", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                  EGL_PLATFORM="surfaceless", LIBGL_ALWAYS_SOFTWARE="1",
                  OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", LP_NUM_THREADS="2")
MESA_VENDOR = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
if not MESA_VENDOR.is_file():
    raise RuntimeError("Local Mesa EGL vendor file is required for verified CPU rendering; no download attempted")
os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(MESA_VENDOR)

import mujoco
import numpy as np
from OpenGL import GL
from PIL import Image, ImageDraw, ImageFont


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="MP4 output path")
    parser.add_argument("--seconds", default=30., type=float)
    parser.add_argument("--fps", default=25, type=int)
    parser.add_argument("--width", default=960, type=int)
    parser.add_argument("--height", default=640, type=int)
    args = parser.parse_args()
    if not 0 < args.seconds <= 60 or not 0 < args.fps <= 50:
        parser.error("Require 0 < seconds <= 60 and 0 < fps <= 50")
    if args.width < 320 or args.height < 240 or args.width % 2 or args.height % 2:
        parser.error("Width/height must be even and at least 320x240")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Local ffmpeg is required; no download attempted")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("schema") != "pera_skateboard_independent_certification_v1":
        raise ValueError("Expected independent skateboard certification report")
    nominal = next(row for row in report["results"] if row["name"] == "nominal")
    trajectory, model_path = Path(nominal["trajectory"]), Path(nominal["model_xml"])
    for path, expected in ((trajectory, nominal["trajectory_sha256"]),
                           (model_path, nominal["model_xml_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"Recorded input changed after evaluation: {path}")
    with np.load(trajectory, allow_pickle=False) as saved:
        qpos, times = saved["qpos"].copy(), saved["time_s"].copy()
    if len(qpos) != len(times) or len(times) < 2 or not np.isfinite(qpos).all():
        raise ValueError("Invalid recorded qpos/time trajectory")
    duration = min(args.seconds, float(times[-1]))
    samples = np.arange(0., duration - 1e-9, 1. / args.fps)
    indices = np.searchsorted(times, samples, side="left").clip(max=len(times)-1)
    # Nearest recorded 50 Hz state for each output frame (exact at default 25 fps).
    previous = np.maximum(indices - 1, 0)
    indices = np.where(np.abs(times[previous] - samples) < np.abs(times[indices] - samples),
                       previous, indices)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if qpos.shape[1] != model.nq:
        raise ValueError("Recorded state dimension differs from saved model")
    model.vis.global_.offwidth, model.vis.global_.offheight = args.width, args.height
    model.vis.quality.shadowsize = 1024
    # Camera-relative lighting keeps the robot and deck visible over the full
    # recorded path. These settings affect rendering only, not saved physics.
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = .55
    model.vis.headlight.diffuse[:] = .8
    model.vis.headlight.specular[:] = .12
    renderer = mujoco.Renderer(model, args.height, args.width)
    gl_renderer = GL.glGetString(GL.GL_RENDERER).decode()
    if not any(name in gl_renderer.lower() for name in ("llvmpipe", "softpipe", "software")):
        renderer.close()
        raise RuntimeError(f"CPU renderer required, got {gl_renderer}")
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.2, 135., -15.
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[:] = 0
    font = ImageFont.truetype("DejaVuSans.ttf", 24)
    small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    attribution = report["source_attribution"]
    comment = ("SuperMimic-PERA learned PyTorch skateboard student; MuJoCo recorded physics; "
               "CPU llvmpipe/libx264. HUSKY geometry/contact/observation adaptations: "
               "TeleHuman/humanoid_skateboarding, CC BY-NC 4.0. "
               f"Checkpoint SHA256 {report['provenance']['checkpoint_sha256']}")
    command = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-threads", "2",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-metadata", "title=supermimic",
               "-metadata", f"comment={comment}",
               str(args.output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    started = time.monotonic()
    try:
        for number, index in enumerate(indices):
            data.qpos[:] = qpos[index]
            data.time = times[index]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3]
            camera.lookat[2] = .55
            renderer.update_scene(data, camera=camera, scene_option=options)
            picture = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 0, args.width, 44), fill=(245, 248, 252))
            draw.text((args.width/2, 21), "supermimic", anchor="mm",
                      font=font, fill=(25, 35, 45))
            draw.rectangle((0, args.height-27, args.width, args.height), fill=(245, 248, 252))
            draw.text((14, args.height-22),
                      "HUSKY scene/contact/observation adaptations: TeleHuman | CC BY-NC 4.0",
                      font=small_font, fill=(75, 85, 95))
            process.stdin.write(np.asarray(picture).tobytes())
            if number in (0, len(indices)//2, len(indices)-1):
                picture.save(args.output.with_name(f"{args.output.stem}_frame_{number:04d}.png"))
            if number % 250 == 0:
                print(f"CPU rendered {number}/{len(indices)} frames", flush=True)
        process.stdin.close()
        process.stdin = None
        _, error = process.communicate()
        if process.returncode:
            raise RuntimeError(f"ffmpeg failed: {error.decode(errors='replace')}")
    finally:
        renderer.close()
        if process.poll() is None:
            process.terminate()
            process.wait()
    summary = {"video": str(args.output.resolve()), "video_sha256": sha256(args.output),
               "title": "supermimic", "headlight": {"ambient": .55, "diffuse": .8, "specular": .12},
               "source_report": str(args.report.resolve()), "report_sha256": sha256(args.report),
               "checkpoint_sha256": report["provenance"]["checkpoint_sha256"],
               "source_trajectory_sha256": nominal["trajectory_sha256"],
               "renderer": gl_renderer, "encoder": "ffmpeg libx264 CPU", "fps": args.fps,
               "frames": len(indices), "video_duration_s": len(indices)/args.fps,
               "source_duration_s": float(times[-1]), "source_nominal_success": nominal["success"],
               "certification_passed_scenarios": report["passed_scenarios"],
               "wall_seconds": time.monotonic()-started, "new_download_bytes": 0,
               "source_attribution": attribution}
    args.output.with_name(f"{args.output.stem}_render.json").write_text(
        json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
