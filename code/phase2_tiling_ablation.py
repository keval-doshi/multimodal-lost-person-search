"""
Phase 2c - Targeted tiling ablation.

PRW contains very few small pedestrians (only 6.8% of ground-truth boxes are
under 96 px, and 0.2% under 64 px), so slicing the whole dataset would spend
an hour to test a technique on frames where it cannot help.

Instead this selects only the frames that DO contain a small person and runs
the tiled detector on those. If tiling fails on the subset most favourable to
it, the negative result is as strong as it can be made.

Step 1 - list the frames worth testing:
    python phase2_tiling_ablation.py --root .\\dataset --select --max-height 96

Step 2 - run tiled detection on just those frames:
    python phase2_detect.py --root .\\dataset --mode tiled \\
        --frame-list ..\\cache\\small_frames.txt

Step 3 - compare, restricted to the same frames and to small boxes:
    python phase2_tiling_ablation.py --root .\\dataset --compare --max-height 96
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from phase2_eval import load_gt, load_split, evaluate, HEIGHT_BINS, BIN_LABELS


def select(prw, cache, max_h, split):
    frames = sorted(f.name for f in (prw / "frames").glob("*.jpg"))
    if split != "all":
        sp = load_split(prw, f"frame_{split}.mat")
        if sp:
            frames = [f for f in frames if f in sp]
            print(f"Split '{split}': {len(frames)} frames")

    gt = load_gt(prw, frames)
    picked, n_small = [], 0
    for f in frames:
        g = gt[f]
        if len(g) == 0:
            continue
        h = g[:, 3] - g[:, 1]
        k = int((h < max_h).sum())
        if k:
            picked.append(f)
            n_small += k

    out = cache / "small_frames.txt"
    out.write_text("\n".join(picked), encoding="utf-8")

    print(f"\nFrames containing a box under {max_h} px: {len(picked)} "
          f"({len(picked) / max(len(frames), 1) * 100:.1f}% of the split)")
    print(f"Small boxes in them: {n_small}")
    cams = sorted({f.split('s')[0] for f in picked})
    print(f"Cameras represented: {', '.join(cams)}")
    print(f"\nWrote {out}")
    print(f"Estimated tiled runtime at 4 fps: {len(picked) / 4 / 60:.0f} min")


def compare(prw, cache, max_h, iou):
    lst = cache / "small_frames.txt"
    if not lst.exists():
        sys.exit("Run --select first.")
    frames = [l.strip() for l in lst.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Comparing on {len(frames)} frames containing small people\n")

    dets = {}
    for mode, fname in (("full", "det_full.parquet"),
                        ("tiled", "det_tiled_small.parquet")):
        p = cache / fname
        if not p.exists():
            p = cache / f"det_{mode}.parquet"
        if not p.exists():
            sys.exit(f"Missing detection cache for '{mode}' ({fname})")
        d = pd.read_parquet(p)
        dets[mode] = d[d["frame"].isin(frames)]
        print(f"{mode:6s} {len(dets[mode]):7d} detections")

    gt_all = load_gt(prw, frames)

    # restrict ground truth to small boxes only - the population tiling targets
    gt_small = {f: g[(g[:, 3] - g[:, 1]) < max_h] for f, g in gt_all.items()}
    n_small = sum(len(g) for g in gt_small.values())

    print(f"\n{'=' * 62}")
    print(f"RECALL ON BOXES UNDER {max_h} px   (n = {n_small})")
    print("=" * 62)
    rows = []
    for mode, d in dets.items():
        _, hit, tp, fp = evaluate(d, gt_small, frames, iou)
        rec = hit.mean() if len(hit) else 0.0
        rows.append({"method": mode, "recall_small": rec,
                     "detections": len(d)})
        print(f"  {mode:6s} recall {rec * 100:5.1f} %   ({int(hit.sum())}/{len(hit)})")

    if len(rows) == 2:
        gain = (rows[1]["recall_small"] - rows[0]["recall_small"]) * 100
        extra = rows[1]["detections"] - rows[0]["detections"]
        print(f"\n  tiling changes small-box recall by {gain:+.1f} pp")
        print(f"  at the cost of {extra:+d} extra detections "
              f"({extra / max(rows[0]['detections'], 1) * 100:+.0f}%)")
        print()
        if gain < 1:
            print("  CONCLUSION: tiling does not improve recall on PRW even on")
            print("  the frames most favourable to it. Report as a negative")
            print("  result: PRW's pedestrians are rarely small enough for")
            print("  slicing-aided inference to contribute.")
        else:
            print("  CONCLUSION: tiling helps on small boxes. Weigh the recall")
            print("  gain against the extra false positives.")

    # full picture across all box sizes on these frames, for the appendix
    print(f"\n{'=' * 62}")
    print("ALL BOX SIZES ON THE SAME FRAMES")
    print("=" * 62)
    tab = {}
    for mode, d in dets.items():
        h, hit, tp, fp = evaluate(d, gt_all, frames, iou)
        idx = np.digitize(h, HEIGHT_BINS) - 1
        col = {}
        for b, lab in enumerate(BIN_LABELS):
            m = idx == b
            if m.any():
                col[lab] = hit[m].mean() * 100
        tab[mode] = col
        prec = tp / (tp + fp) if (tp + fp) else 0
        print(f"  {mode:6s} overall recall {hit.mean() * 100:5.1f} %   "
              f"precision {prec * 100:5.1f} %")
    df = pd.DataFrame(tab).round(1)
    if len(df.columns) == 2:
        df["gain (pp)"] = (df["tiled"] - df["full"]).round(1)
    print()
    print(df.to_string())

    out = cache / "phase2_tiling_ablation.csv"
    df.to_csv(out)
    print(f"\nSaved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--max-height", type=float, default=96)
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    root = Path(a.root).resolve()
    prw = root / "PRW"
    cache = root.parent / "cache"
    cache.mkdir(exist_ok=True)

    if a.select:
        select(prw, cache, a.max_height, a.split)
    elif a.compare:
        compare(prw, cache, a.max_height, a.iou)
    else:
        ap.error("pass --select or --compare")


if __name__ == "__main__":
    main()
