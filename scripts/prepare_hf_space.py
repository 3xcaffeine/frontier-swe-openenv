"""Assemble the push payload for an HF Space.

Given a task name, produce a directory that can be force-pushed to the Space:
- Dockerfile and README.md are lifted from ``spaces/<task>/`` to the payload root
  (HF requires both at the root for Docker Spaces).
- The sibling ``spaces/<other-task>/`` subtree is dropped to reduce Space size.
- ``.gitattributes`` is preserved so HF correctly interprets the LFS-tracked
  bundle.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TASKS = ("notebook", "postgres")


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

    for name in ("Dockerfile", "README.md"):
        src = space_src / name
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        shutil.copy2(src, out / name)

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
