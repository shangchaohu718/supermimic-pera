#!/usr/bin/env python3
"""Prepare offline wheels from complete uv cache entries; never download packages.

Run with the Python ABI that will be used by .venv-local. PyTorch, CUDA and
Triton remain inherited from that interpreter's existing environment.
"""

from __future__ import annotations

import argparse
from importlib.metadata import distributions
import os
from pathlib import Path
import subprocess
from zipfile import ZIP_STORED, ZipFile, is_zipfile

from packaging.requirements import Requirement
from packaging.tags import parse_tag, sys_tags
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/supermimic_wheels")
    parser.add_argument("--uv-cache", type=Path,
                        default=Path(os.environ.get("UV_CACHE_DIR", Path.home() / ".cache/uv")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    exported = subprocess.check_output([
        "uv", "export", "--offline", "--locked", "--no-dev", "--no-hashes",
        "--no-header", "--no-annotate", "--no-emit-project", "--format", "requirements-txt",
    ], cwd=ROOT, text=True)
    requirements = [Requirement(line) for line in exported.splitlines() if line.strip()]
    requirements = [r for r in requirements if not r.marker or r.marker.evaluate()]
    requirements += [Requirement("pytest"), Requirement("pluggy"), Requirement("iniconfig")]
    requirements += [Requirement("uv-build>=0.12.1,<0.13")]
    supported = set(sys_tags())
    cached = {}
    for source in (args.uv_cache / "archive-v0").iterdir():
        for dist in distributions(path=[str(source)]):
            wheel = next(source.glob("*.dist-info/WHEEL"), None)
            if wheel is None:
                continue
            tags = [line[5:].strip() for line in wheel.read_text().splitlines()
                    if line.startswith("Tag: ")]
            compatible = [tag for tag in tags if parse_tag(tag) & supported]
            if compatible:
                cached[(canonicalize_name(dist.metadata["Name"]), Version(dist.version))] = (
                    source, compatible[0])
    plan, missing = [], []
    for requirement in requirements:
        name = canonicalize_name(requirement.name)
        if name.startswith(("nvidia-", "cuda-")) or name in {"torch", "triton"}:
            continue
        candidates = [(key, value) for key, value in cached.items()
                      if key[0] == name and (not requirement.specifier or key[1] in requirement.specifier)]
        if not candidates:
            missing.append(str(requirement))
            continue
        (name, version), (source, tag) = max(candidates, key=lambda item: item[0][1])
        target = args.output / f"{name.replace('-', '_')}-{version}-{tag}.whl"
        plan.append((source, target))
    if missing:
        raise SystemExit("Missing complete, ABI-compatible local cache entries:\n  "
                         + "\n  ".join(missing)
                         + "\nNo packages were downloaded. Obtain permission before fetching missing files.")
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        for source, target in plan:
            if target.is_file() and is_zipfile(target):
                continue
            temporary = target.with_suffix(".whl.tmp")
            with ZipFile(temporary, "w", compression=ZIP_STORED) as archive:
                for path in source.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(source))
            temporary.replace(target)
    print(f"{'Validated' if args.dry_run else 'Prepared'} {len(plan)} local wheels; downloaded 0 bytes")


if __name__ == "__main__":
    main()
