#!/usr/bin/env bash
# Run G1 SuperMimic with local wheels and the existing GMR PyTorch/CUDA runtime.
set -euo pipefail

# Downloads require the user's explicit permission; all default paths are offline.
export UV_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "$project_root/src/unisupermimic" ]]; then
  source_root="$project_root"
else
  source_root="$project_root/third_party/Uni-SuperMimic"
fi
env_root="$source_root/.venv-local"
command_name="${1:-help}"
if (($#)); then shift; fi

usage() {
  cat <<'EOF'
Usage: tools/run_supermimic.sh COMMAND [arguments]

  setup                 Prepare a local overlay using cached wheels and GMR
  assets                Validate existing native G1/LAFAN1 assets offline
  validate-assets       Validate assets offline
  test [pytest args]    Run the upstream non-slow tests
  smoke [backend]       Run two tiny training iterations (default: mjwarp)
  train [Hydra args]    Run the native training entry point
  eval [args]           Run the native checkpoint evaluator
  video [args]          Run the native video exporter

Backend choices for smoke: mjwarp, mujoco, motrix.
Extra smoke arguments are Hydra overrides. Smoke uses the native network widths,
four environments, and short updates; its checkpoint is only a wiring check.
SUPERMIMIC_PYTHON selects an existing Python/PyTorch base for first-time setup.
SUPERMIMIC_WHEELHOUSE selects the offline wheel directory.
All commands are offline. Missing source, assets, or packages cause an error;
they are never downloaded automatically by this wrapper.
EOF
}

case "$command_name" in
  help|-h|--help) usage; exit 0 ;;
  setup|assets|validate-assets|test|smoke|train|eval|video) ;;
  *) usage >&2; exit 2 ;;
esac

if [[ "$command_name" == setup ]]; then
  # Do not reset an existing checkout: it may contain ongoing G1 research.
  if [[ ! -f "$source_root/pyproject.toml" ]]; then
    echo 'SuperMimic source is missing; downloading it requires explicit permission.' >&2
    exit 1
  fi
  wheel_root="${SUPERMIMIC_WHEELHOUSE:-$project_root/build/supermimic_wheels}"
  if [[ ! -d "$wheel_root" ]]; then
    echo 'Local wheel cache is missing; restore it before offline setup.' >&2
    exit 1
  fi
  if [[ ! -x "$env_root/bin/python" ]]; then
    if [[ -z "${SUPERMIMIC_PYTHON:-}" ]]; then
      echo 'Set SUPERMIMIC_PYTHON to an existing Python/PyTorch environment.' >&2
      exit 1
    fi
    uv venv --offline --no-python-downloads --system-site-packages \
      --python "$SUPERMIMIC_PYTHON" "$env_root"
  fi
  uv pip install --offline --no-deps --python "$env_root/bin/python" "$wheel_root"/*.whl
  cd -- "$source_root"
  exec uv pip install --offline --no-deps --no-build-isolation \
    --python "$env_root/bin/python" --editable . "$@"
fi

if [[ ! -f "$source_root/pyproject.toml" ]]; then
  echo 'SuperMimic is missing. Run tools/run_supermimic.sh setup first.' >&2
  exit 1
fi
if [[ ! -x "$env_root/bin/unisupermimic-train" ]]; then
  echo 'Local SuperMimic environment is missing. Run tools/run_supermimic.sh setup.' >&2
  exit 1
fi
cd -- "$source_root"

case "$command_name" in
  assets|validate-assets)
    exec "$env_root/bin/python" tools/prepare_supermimic_assets.py --validate-only "$@"
    ;;
  test) exec "$env_root/bin/python" -m pytest "$@" ;;
  train) exec "$env_root/bin/unisupermimic-train" "$@" ;;
  eval) exec "$env_root/bin/unisupermimic-eval" "$@" ;;
  video) exec "$env_root/bin/unisupermimic-video" "$@" ;;
  smoke)
    backend="${1:-mjwarp}"
    if (($#)); then shift; fi
    case "$backend" in
      mjwarp|mujoco|motrix) ;;
      *) echo "Unknown backend: $backend" >&2; exit 2 ;;
    esac
    mkdir -p "$project_root/runs"
    smoke_dir="$(mktemp -d "$project_root/runs/g1_supermimic_smoke_${backend}_XXXXXX")"
    echo "SuperMimic smoke output: $smoke_dir"
    exec "$env_root/bin/unisupermimic-train" \
      "task=g1_super_mimic/$backend" \
      algo.num_envs=4 algo.iterations=2 algo.buffer_steps=64 \
      algo.warmup_steps=16 algo.wm_batches=1 algo.wm_batch=8 \
      algo.pi_batch=8 algo.horizon=4 algo.wm_only_iters=1 \
      algo.critic_epochs=1 algo.critic_minibatches=1 \
      algo.compile=false algo.log_every=1 algo.ckpt_every=2 \
      training.no_play=true "training.log_dir=$smoke_dir" "$@"
    ;;
esac
