#!/usr/bin/env python3
"""Launch and inspect a persistent offline native SuperMimic training run.

The supervisor only manages native entry points, checkpoints and evaluations.
It does not replace the SuperMimic learner or its training algorithm.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT if (ROOT / "src/unisupermimic").is_dir() else ROOT / "third_party/Uni-SuperMimic"
WRAPPER = ROOT / "tools/run_supermimic.sh"
PROFILE = ROOT / "configs/g1_supermimic_baseline.json"
DEFAULT_RUN = ROOT / "runs/g1_supermimic_baseline_20261023_seed0"


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def latest_metrics(run: Path) -> dict:
    path = run / "train_log.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if all(v is not None for v in r.values())]
    return rows[-1] if rows else {}


def local_environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        UV_OFFLINE="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        WANDB_MODE="offline", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0",
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4",
        TORCHINDUCTOR_COMPILE_THREADS="1", MAX_JOBS="1",
    )
    return env


def run_worker(run: Path, resume: bool) -> int:
    profile = json.loads((run / "profile.json").read_text())
    with (run / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        overrides = profile["hydra_overrides"]
        command = ["bash", str(WRAPPER), "train", *overrides, f"training.log_dir={run}"]
        if resume:
            command += ["algo.resume=true", f"algo.load_run={run}", "algo.checkpoint=last.pt"]
        state = {
            "state": "starting", "supervisor_pid": os.getpid(), "started_at": now(),
            "run_dir": str(run), "command": command, "resume": resume,
            "evaluation_results": {}, "download_bytes_this_run": 0,
        }
        stop_requested = False

        def request_stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        train_log = (run / ("resume_console.log" if resume else "console.log")).open("a")
        training = subprocess.Popen(command, cwd=ROOT, env=local_environment(),
                                    stdout=train_log, stderr=subprocess.STDOUT)
        state.update(state="training", train_pid=training.pid)
        write_json(run / "status.json", state)
        print(f"{now()} Training started: PID {training.pid}", flush=True)
        evaluation = None
        eval_log = None
        evaluating = None
        attempted = set()
        for iteration in profile["evaluation"]["iterations"]:
            result = run / f"ckpt_{iteration:06d}_eval.json"
            if result.exists():
                state["evaluation_results"][str(iteration)] = json.loads(result.read_text())
                attempted.add(iteration)
        try:
            while True:
                metrics = latest_metrics(run)
                state.update(updated_at=now(), latest_metrics=metrics,
                             training_exit_code=training.poll())
                if metrics and any(not math.isfinite(float(v)) for v in metrics.values()):
                    state.update(state="failed", error="Non-finite training metric")
                    stop_requested = True
                if evaluation is not None and evaluation.poll() is not None:
                    code = evaluation.returncode
                    result = run / f"ckpt_{evaluating:06d}_eval.json"
                    if code == 0 and result.is_file():
                        state["evaluation_results"][str(evaluating)] = json.loads(result.read_text())
                    else:
                        state["evaluation_results"][str(evaluating)] = {
                            "error": "Native evaluation failed", "exit_code": code}
                    print(f"{now()} Evaluation {evaluating} finished: exit {code}", flush=True)
                    eval_log.close()
                    evaluation, evaluating = None, None
                if stop_requested:
                    if training.poll() is None:
                        training.terminate()
                    if evaluation is not None:
                        evaluation.terminate()
                    state["state"] = "failed" if state.get("error") else "stopped"
                    write_json(run / "status.json", state)
                    return 1 if state.get("error") else 130
                if evaluation is None:
                    for iteration in profile["evaluation"]["iterations"]:
                        checkpoint = run / f"ckpt_{iteration:06d}.pt"
                        if iteration in attempted or not checkpoint.is_file():
                            continue
                        attempted.add(iteration)
                        evaluating = iteration
                        eval_command = ["bash", str(WRAPPER), "train", *overrides,
                                        "training.play_only=true", f"algo.load_run={run}",
                                        f"algo.checkpoint={checkpoint.name}"]
                        eval_log = (run / f"eval_{iteration:06d}.log").open("a")
                        evaluation = subprocess.Popen(eval_command, cwd=ROOT,
                                                      env=local_environment(), stdout=eval_log,
                                                      stderr=subprocess.STDOUT)
                        print(f"{now()} Evaluating checkpoint {iteration}: PID {evaluation.pid}", flush=True)
                        break
                state["evaluating_iteration"] = evaluating
                write_json(run / "status.json", state)
                if training.poll() is not None and evaluation is None:
                    state.update(state="completed" if training.returncode == 0 else "failed",
                                 training_exit_code=training.returncode, finished_at=now())
                    if any("error" in v for v in state["evaluation_results"].values()):
                        state["state"] = "completed_with_evaluation_errors" if training.returncode == 0 else "failed"
                    write_json(run / "status.json", state)
                    return training.returncode
                time.sleep(10)
        except Exception as error:
            state.update(state="failed", error=f"{type(error).__name__}: {error}",
                         updated_at=now())
            write_json(run / "status.json", state)
            raise
        finally:
            if training.poll() is None:
                training.terminate()
                try:
                    training.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    training.kill()
            if evaluation is not None and evaluation.poll() is None:
                evaluation.terminate()
            if eval_log is not None:
                eval_log.close()
            train_log.close()


def start(run: Path, resume: bool) -> None:
    run.mkdir(parents=True, exist_ok=True)
    existing = run / "status.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        pid = old.get("supervisor_pid")
        if pid and Path(f"/proc/{pid}/cmdline").exists():
            if b"g1_supermimic_baseline.py" in Path(f"/proc/{pid}/cmdline").read_bytes():
                raise RuntimeError(f"This run already has supervisor PID {pid}")
    if resume and not (run / "last.pt").is_file():
        raise FileNotFoundError("Resume requires the full last.pt checkpoint")
    if not resume and ((run / "train_log.csv").exists() or (run / "last.pt").exists()):
        raise RuntimeError("Existing training data found; use --resume or choose a new run directory")
    if not resume:
        profile = json.loads(PROFILE.read_text())
        observed_commit = subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
        ancestry = subprocess.run(
            ["git", "-C", str(SOURCE), "merge-base", "--is-ancestor",
             profile["upstream_commit"], observed_commit], check=False)
        if ancestry.returncode != 0:
            raise RuntimeError(f"Expected upstream ancestry missing from: {observed_commit}")
        write_json(run / "profile.json", profile)
        asset_root = SOURCE / "src/unisupermimic/assets"
        motions = sorted((asset_root / "motions/g1/lafan1_supermimic").glob("*.csv"))
        if len(motions) != 40:
            raise RuntimeError(f"Expected 40 local motion clips; found {len(motions)}")
        files = list(motions)
        files += [asset_root / "robots/g1/supermimic.xml", asset_root / "robots/g1/scene_supermimic.xml"]
        files += sorted((asset_root / "robots/g1/assets").glob("*.STL"))
        write_json(run / "assets_manifest.json", {
            "created_at": now(), "motion_clips": len(motions),
            "sha256": {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        })
        write_json(run / "provenance.json", {
            "created_at": now(), "upstream_commit": profile["upstream_commit"],
            "source_commit": observed_commit,
            "environment_setup": str(ROOT / "reports/g1_supermimic_setup/summary.json"),
            "initialization": "random native networks; no pretrained G1 checkpoint available",
            "data": "40 locally available LAFAN1/G1 clips, 264705 source frames",
            "downloads_this_run_bytes": 0,
            "resume_semantics": "Native full last.pt; replay is rebuilt, not an exact trajectory continuation",
        })
    with (run / "supervisor.log").open("a") as output:
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--run-dir", str(run)]
        if resume:
            command.append("--resume")
        process = subprocess.Popen(command, cwd=ROOT, env=local_environment(),
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"supervisor_pid": process.pid, "run_dir": str(run)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "status", "worker"])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    if args.action == "start":
        start(run, args.resume)
    elif args.action == "worker":
        raise SystemExit(run_worker(run, args.resume))
    else:
        state = json.loads((run / "status.json").read_text())
        state["latest_metrics"] = latest_metrics(run)
        pid = state.get("supervisor_pid")
        state["supervisor_alive"] = bool(pid and Path(f"/proc/{pid}/cmdline").exists()
                                         and b"g1_supermimic_baseline.py" in Path(f"/proc/{pid}/cmdline").read_bytes())
        state["evaluation_results"] = {
            k: {field: v[field] for field in ("norm_return", "norm_return_framew", "n_failed", "error", "exit_code") if field in v}
            for k, v in state.get("evaluation_results", {}).items()
        }
        print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
