"""
Phase 2b - Evaluate detections against PRW ground truth.

Produces the O2 result: recall as a function of ground-truth box height,
comparing whole-frame inference against tiled inference.

  python phase2_eval.py --root .\\dataset

Reads cache/det_full.parquet and cache/det_tiled.parquet, writes
cache/phase2_results.csv and cache/phase2_recall_by_height.png

Ground truth notes (from the PRW readme):
  - annotations/*.mat holds one row per pedestrian box
  - coordinates are [x, y, w, h]; IDs run 1..932 plus -2
  - -2 means "identity unknown" - excluded from re-ID but VALID for
    detection, so all boxes are used here
  - frame_test.mat / frame_train.mat give the fixed split
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

HEIGHT_BINS = [0, 32, 64, 96, 128, 192, 256, 10 ** 6]
BIN_LABELS = ["<32", "32-64", "64-96", "96-128", "128-192", "192-256", "256+"]


# ------------------------------------------------------------ gt loading
def first_array(mat):
    """Return the first non-metadata array in a loadmat dict."""
    for k, v in mat.items():
        if not k.startswith("__") and isinstance(v, np.ndarray) and v.size:
            return v
    return None


def detect_column_order(sample_rows, W=1920, H=1080):
    """
    PRW annotation rows have 5 columns: an ID and [x, y, w, h].
    Mirrors differ on whether the ID comes first or last, so decide from
    the data: score each interpretation by how many boxes are geometrically
    valid (positive size, inside the frame) and how ID-like the other column
    looks. Geometry is the strong signal; the ID range alone can tie.
    """
    a = np.vstack(sample_rows).astype(float)

    def geom_score(xywh):
        x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        ok = (w > 0) & (h > 0) & (x >= -2) & (y >= -2) \
             & (x + w <= W + 2) & (y + h <= H + 2)
        return ok.mean()

    def id_score(col):
        return (((col >= -2) & (col <= 932))
                & (np.abs(col - np.round(col)) < 1e-6)).mean()

    s_first = geom_score(a[:, 1:5]) + 0.25 * id_score(a[:, 0])
    s_last = geom_score(a[:, 0:4]) + 0.25 * id_score(a[:, 4])
    return "id_first" if s_first >= s_last else "id_last"


def load_gt(prw, frames):
    ann_dir = prw / "annotations"
    sample = []
    for f in frames[:50]:
        m = loadmat(str(ann_dir / (f + ".mat")))
        a = first_array(m)
        if a is not None and a.ndim == 2 and a.shape[1] >= 5:
            sample.append(np.atleast_2d(a)[:, :5])
    if not sample:
        sys.exit("ERROR: could not read annotation .mat files.")

    order = detect_column_order(sample)
    print(f"Annotation column order detected: {order}")

    gt = {}
    n_box = 0
    for f in frames:
        p = ann_dir / (f + ".mat")
        if not p.exists():
            gt[f] = np.empty((0, 4))
            continue
        a = first_array(loadmat(str(p)))
        if a is None or a.ndim != 2 or a.shape[1] < 5:
            gt[f] = np.empty((0, 4))
            continue
        a = np.atleast_2d(a).astype(float)
        xywh = a[:, 1:5] if order == "id_first" else a[:, 0:4]
        x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        gt[f] = np.stack([x, y, x + w, y + h], axis=1)
        n_box += len(gt[f])
    print(f"Ground truth: {n_box} boxes over {len(frames)} frames")
    return gt


def load_split(prw, name):
    p = prw / name
    if not p.exists():
        return None
    a = first_array(loadmat(str(p)))
    out = []
    for v in np.ravel(a):
        s = str(v[0]) if isinstance(v, np.ndarray) and v.size else str(v)
        s = s.strip()
        out.append(s if s.endswith(".jpg") else s + ".jpg")
    return set(out)


# ------------------------------------------------------------ matching
def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    ar_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ar_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (ar_a[:, None] + ar_b[None, :] - inter + 1e-9)


def evaluate(det_df, gt, frames, iou_thr):
    """Greedy match by descending confidence. Returns per-GT hit flags."""
    by_frame = {f: g.sort_values("conf", ascending=False)
                for f, g in det_df.groupby("frame")} if len(det_df) else {}

    heights, hits = [], []
    tp = fp = 0
    for f in frames:
        g = gt.get(f, np.empty((0, 4)))
        d = by_frame.get(f)
        db = d[["x1", "y1", "x2", "y2"]].to_numpy() if d is not None else np.empty((0, 4))

        matched_gt = np.zeros(len(g), bool)
        matched_det = np.zeros(len(db), bool)
        if len(g) and len(db):
            M = iou_matrix(db, g)
            for di in range(len(db)):                # already conf-sorted
                gi = int(np.argmax(M[di]))
                if M[di, gi] >= iou_thr and not matched_gt[gi]:
                    matched_gt[gi] = True
                    matched_det[di] = True
        tp += int(matched_det.sum())
        fp += int(len(db) - matched_det.sum())
        heights.extend((g[:, 3] - g[:, 1]).tolist())
        hits.extend(matched_gt.tolist())

    return np.array(heights), np.array(hits), tp, fp


def summarise(name, heights, hits, tp, fp):
    n = len(heights)
    rec = hits.mean() if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    print(f"\n{name}")
    print(f"  overall recall    {rec * 100:5.1f} %   ({int(hits.sum())}/{n})")
    print(f"  overall precision {prec * 100:5.1f} %   ({tp} tp, {fp} fp)")

    idx = np.digitize(heights, HEIGHT_BINS) - 1
    rows = []
    print(f"  {'height (px)':<12}{'n':>8}{'recall':>10}")
    for b, lab in enumerate(BIN_LABELS):
        m = idx == b
        if not m.any():
            continue
        r = hits[m].mean()
        rows.append({"method": name, "bin": lab, "n": int(m.sum()),
                     "recall": r})
        print(f"  {lab:<12}{int(m.sum()):>8}{r * 100:>9.1f}%")
    return rows, rec, prec


def profile_gt(prw, limit=0):
    """Report the ground-truth box-height distribution across the dataset.

    This decides whether tiling can help at all: if PRW contains very few
    small boxes, slicing has nothing to recover and will only cost precision.
    """
    frames = sorted(f.name for f in (prw / "frames").glob("*.jpg"))
    if limit:
        frames = frames[:limit]
    gt = load_gt(prw, frames)
    h = np.concatenate([g[:, 3] - g[:, 1] for g in gt.values() if len(g)])
    print(f"\nGround-truth box heights over {len(frames)} frames "
          f"({len(h)} boxes)")
    for q in (5, 10, 25, 50, 75, 90, 95):
        print(f"  p{q:<3} {np.percentile(h, q):7.0f} px")
    print()
    idx = np.digitize(h, HEIGHT_BINS) - 1
    for b, lab in enumerate(BIN_LABELS):
        n = int((idx == b).sum())
        if n:
            print(f"  {lab:<10}{n:>8}  {n / len(h) * 100:5.1f}%")
    small = (h < 96).mean() * 100
    print(f"\nBoxes under 96 px: {small:.1f}%")
    if small < 10:
        print("=> Very few small people. Tiling is unlikely to help on PRW,")
        print("   and that is itself a reportable finding.")
    else:
        print("=> Enough small people for tiling to matter.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--profile-gt", action="store_true",
                    help="report ground-truth box sizes only; no detections needed")
    ap.add_argument("--profile-limit", type=int, default=0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--split", default="test",
                    choices=["test", "train", "all"])
    ap.add_argument("--smoke", action="store_true",
                    help="evaluate the *_smoke caches instead")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    prw = root / "PRW"
    cache = root.parent / "cache"

    if a.profile_gt:
        profile_gt(prw, a.profile_limit)
        return
    suffix = "_smoke" if a.smoke else ""

    dets = {}
    for mode in ("full", "tiled"):
        p = cache / f"det_{mode}{suffix}.parquet"
        if p.exists():
            dets[mode] = pd.read_parquet(p)
            print(f"Loaded {p.name}: {len(dets[mode])} detections")
        else:
            print(f"(missing {p.name} - skipping {mode})")
    if not dets:
        sys.exit("No detection caches found. Run phase2_detect.py first.")

    frames = sorted(set().union(*[set(d["frame"].unique()) for d in dets.values()]))

    if a.split != "all" and not a.smoke:
        sp = load_split(prw, f"frame_{a.split}.mat")
        if sp:
            before = len(frames)
            frames = [f for f in frames if f in sp]
            print(f"Split '{a.split}': {len(frames)} of {before} frames")
        else:
            print(f"(frame_{a.split}.mat unreadable - using all frames)")

    gt = load_gt(prw, frames)

    all_rows, summary = [], []
    for mode, df in dets.items():
        df = df[df["frame"].isin(frames)]
        h, hit, tp, fp = evaluate(df, gt, frames, a.iou)
        rows, rec, prec = summarise(mode, h, hit, tp, fp)
        all_rows += rows
        summary.append({"method": mode, "recall": rec, "precision": prec,
                        "detections": len(df)})

    res = pd.DataFrame(all_rows)
    out_csv = cache / f"phase2_results{suffix}.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    if {"full", "tiled"} <= set(dets):
        print("\n" + "=" * 62)
        print("TILED vs FULL - recall gain by ground-truth box height")
        print("=" * 62)
        piv = res.pivot(index="bin", columns="method", values="recall")
        piv = piv.reindex([b for b in BIN_LABELS if b in piv.index])
        out = pd.DataFrame({
            "full %": (piv["full"] * 100).round(1),
            "tiled %": (piv["tiled"] * 100).round(1),
            "gain (pp)": ((piv["tiled"] - piv["full"]) * 100).round(1),
        })
        print(out.to_string())
        print("\nThis table is your O2 result.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for mode in dets:
            sub = res[res["method"] == mode].set_index("bin")
            sub = sub.reindex([b for b in BIN_LABELS if b in sub.index])
            ax.plot(sub.index, sub["recall"] * 100, marker="o", label=mode)
        ax.set_xlabel("ground-truth box height (px)")
        ax.set_ylabel("recall (%)")
        ax.set_title("Detection recall by person size")
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        png = cache / f"phase2_recall_by_height{suffix}.png"
        fig.savefig(png, dpi=150)
        print(f"Saved {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
