#!/usr/bin/env python3
"""
Build a minimal ViPE-Cambrians lookup JSON from the extracted Cambrian-P-Data tree.

The loader reads this lookup to map every training-jsonl video stem onto its
pose `.npz`. We don't ship the (large) FAIR-internal lookup on HuggingFace;
users regenerate it locally from the data they already have after extracting
`pose/*.tar` from `nyu-visionx/Cambrian-P-Data`.

INPUT — required directory structure (produced by
`huggingface-cli download nyu-visionx/Cambrian-P-Data --local-dir $VIPE_CAMBRIANS_RESULTS_ROOT`
followed by `for t in pose/*.tar; do tar xf "$t"; done` per README Quickstart):

    $VIPE_CAMBRIANS_RESULTS_ROOT/
    └── results/
        ├── result_0/
        │   └── <source>/                   # one of the 10 kept sources
        │       └── <uid>/                  # one Cambrian-S clip uid
        │           └── pose/
        │               └── <uid>.npz       # cam2world matrices, shape (N,4,4) or (N,3,4)
        ├── result_1/
        └── result_N/                       # 35,802 .npz total across all result_*/<source>/<uid>/

10 expected sources (`data/vipe_source_config_high.json`): Ego4d, favd, guiworld,
k400_targz, k710, sharegpt4o, ssv2, star, tgif, timeit.

OUTPUT — one JSON list, ~5-10 MB. Each entry:

    {
        "video":             "<source>/<uid>.mp4",        # RGB stem; matched against Cambrian-S 3M jsonl
        "pose_path":         "results/result_X/<source>/<uid>/pose/<uid>.npz",  # relative to --results_root
        "source":            "<source>",
        "max_velocity":      float,  # max ||t[i+1] - t[i]|| in metric scale of the .npz
        "max_rotation_rate": float,  # max angle between R[i+1] and R[i] in degrees, frame-to-frame
    }

Usage (paths match README §Data Preparation Quickstart):

    python scripts/data/build_vipe_lookup.py \\
        --results_root $VIPE_CAMBRIANS_RESULTS_ROOT \\
        --out          $VIPE_CAMBRIANS_RESULTS_ROOT/vipe_cambrians_with_vqa.json

Then the Quickstart's existing symlink line works as-is:
    ln -sf "$VIPE_CAMBRIANS_RESULTS_ROOT/vipe_cambrians_with_vqa.json" data/vipe_cambrians_with_vqa.json
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np


# Same priority list as load_vipe_cambrians_poses in vipe_dataloading_utils.py:362
POSE_KEYS = ("data", "poses", "poses_pred", "camera_poses")


def load_c2w(npz_path):
    """Read cam2world (N,4,4) from a .npz, normalizing (N,3,4) → (N,4,4)."""
    with np.load(npz_path, allow_pickle=True) as data:
        for k in POSE_KEYS:
            if k in data.files:
                p = data[k].astype(np.float32)
                break
        else:
            p = data[data.files[0]].astype(np.float32)
    if p.ndim == 3 and p.shape[1:] == (3, 4):
        p4 = np.zeros((len(p), 4, 4), dtype=np.float32)
        p4[:, :3, :] = p
        p4[:, 3, 3] = 1.0
        p = p4
    return p


def max_velocity(c2w):
    if len(c2w) < 2:
        return 0.0
    t = c2w[:, :3, 3]
    return float(np.linalg.norm(np.diff(t, axis=0), axis=1).max())


def max_rotation_rate_deg(c2w):
    """Max frame-to-frame rotation angle (degrees) over the clip."""
    if len(c2w) < 2:
        return 0.0
    R = c2w[:, :3, :3]
    # R_rel[i] = R[i+1] @ R[i]^T; trace gives 1 + 2 cos(angle).
    R_rel = np.einsum("nij,njk->nik", R[1:], R[:-1].transpose(0, 2, 1))
    traces = np.trace(R_rel, axis1=1, axis2=2)
    cos = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True,
                    help="$VIPE_CAMBRIANS_RESULTS_ROOT — the dir holding results/result_*/...")
    ap.add_argument("--out", required=True,
                    help="Output lookup JSON path.")
    ap.add_argument("--no_stats", action="store_true",
                    help="Skip max_velocity/max_rotation_rate — output is loader-functional "
                         "but lacks filter knobs. Faster.")
    args = ap.parse_args()

    results_dir = osp.join(args.results_root, "results")
    if not osp.isdir(results_dir):
        print(f"ERROR: {results_dir} does not exist. Did you extract pose/*.tar?", file=sys.stderr)
        return 2

    entries = []
    n_seen = n_unreadable = 0
    for result_x in sorted(os.listdir(results_dir)):
        result_x_dir = osp.join(results_dir, result_x)
        if not osp.isdir(result_x_dir):
            continue
        for source in sorted(os.listdir(result_x_dir)):
            source_dir = osp.join(result_x_dir, source)
            if not osp.isdir(source_dir):
                continue
            for uid in sorted(os.listdir(source_dir)):
                pose_npz = osp.join(source_dir, uid, "pose", f"{uid}.npz")
                if not osp.isfile(pose_npz):
                    continue
                n_seen += 1
                e = {
                    "video": f"{source}/{uid}.mp4",
                    "pose_path": osp.relpath(pose_npz, args.results_root),
                    "source": source,
                }
                if not args.no_stats:
                    try:
                        poses = load_c2w(pose_npz)
                        e["max_velocity"] = max_velocity(poses)
                        e["max_rotation_rate"] = max_rotation_rate_deg(poses)
                    except Exception as exc:
                        n_unreadable += 1
                        print(f"[warn] {pose_npz}: {exc}", file=sys.stderr)
                entries.append(e)
                if n_seen % 1000 == 0:
                    print(f"  scanned {n_seen:,} .npz...", file=sys.stderr)

    tmp = args.out + ".tmp"
    os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(entries, f)
    os.replace(tmp, args.out)

    print(f"[done] {len(entries):,} entries → {args.out}", file=sys.stderr)
    if n_unreadable:
        print(f"[warn] {n_unreadable} .npz files couldn't be parsed; entries written without stats", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
