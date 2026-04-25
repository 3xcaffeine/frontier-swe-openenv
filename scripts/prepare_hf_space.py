"""Assemble the push payload for an HF Space.

Given a task name, produce a directory that can be force-pushed to the Space:
- Dockerfile, README.md, and openenv.yaml are lifted from ``spaces/<task>/``
  to the payload root (HF requires Dockerfile + README at the root for Docker
  Spaces; openenv.yaml goes there so judges pulling the Space see a valid
  manifest at the URL root).
- The sibling ``spaces/<other-task>/`` subtree is dropped to reduce Space size.
- ``.gitattributes`` is preserved so HF correctly interprets the LFS-tracked
  bundle.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TASKS = ("notebook", "postgres", "type-checker")


def prepare(task: str, out: Path, repo_root: Path) -> None:
    if task not in TASKS:
        raise SystemExit(f"unknown task {task!r}; expected one of {TASKS}")

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(
        repo_root,
        out,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )

    space_src = out / "spaces" / task
    if not space_src.is_dir():
        raise SystemExit(f"expected {space_src} to exist")

    # Required: HF Docker Spaces need Dockerfile + README.md at the root.
    for name in ("Dockerfile", "README.md"):
        src = space_src / name
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        shutil.copy2(src, out / name)

    # Optional: lift openenv.yaml to the root so judges pulling the Space URL
    # see a valid OpenEnv manifest at the top level. Missing is non-fatal.
    manifest = space_src / "openenv.yaml"
    if manifest.is_file():
        shutil.copy2(manifest, out / "openenv.yaml")

    shutil.rmtree(out / "spaces")
    print(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    prepare(args.task, args.out.resolve(), args.repo_root.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
