#!/usr/bin/env python3
"""Deterministically carve a 'visible' corpus out of the hidden test bundle.

The upstream Harbor orchestrator bind-mounts a visible corpus at
$DATA_ROOT/visible/ from a separate dataset volume. Our standalone
image has no such orchestrator, so we synthesize the visible corpus
at image-build time by taking a seeded random subset of the hidden
bundle's notebook files.

See decision-log D-009 for the rationale (and the reward-hacking
caveat that visible ⊂ hidden).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="Path to hidden_test_set_bundle.zip")
    parser.add_argument("--out", required=True, help="Output directory for visible corpus")
    parser.add_argument("--manifest", required=True, help="Output path for manifest.json")
    parser.add_argument("--ratio", type=float, default=0.75, help="Fraction of files in the visible split")
    parser.add_argument("--seed", type=int, default=17, help="Deterministic shuffle seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    bundle = Path(args.bundle)
    if not bundle.is_file():
        print(f"ERROR: bundle not found: {bundle}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    manifest_path = Path(args.manifest)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nbc_split_") as tmpdir:
        tmp = Path(tmpdir)
        with zipfile.ZipFile(bundle) as zf:
            zf.extractall(tmp)

        files_root = tmp / "hidden_test_set_bundle" / "files"
        if not files_root.is_dir():
            print(
                f"ERROR: bundle is missing hidden_test_set_bundle/files/: {files_root}",
                file=sys.stderr,
            )
            return 2

        all_files = sorted(p for p in files_root.iterdir() if p.is_file())
        if not all_files:
            print("ERROR: no files in bundle", file=sys.stderr)
            return 2

        rng = random.Random(args.seed)
        shuffled = list(all_files)
        rng.shuffle(shuffled)
        n_visible = max(1, int(round(len(shuffled) * args.ratio)))
        visible = shuffled[:n_visible]

        for src in visible:
            shutil.copy2(src, out_dir / src.name)

    manifest = {
        "corpus": "notebook-compression-visible",
        "source_bundle": bundle.name,
        "ratio": args.ratio,
        "seed": args.seed,
        "count": n_visible,
        "files": sorted(p.name for p in visible),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {n_visible} files to {out_dir} and manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
