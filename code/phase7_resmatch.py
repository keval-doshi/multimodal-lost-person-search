"""
Phase 7b - Resolution-matched matching.

The Phase 4 diagnostic found something unplanned: degrading the QUERY to the
same level as the gallery improved Rank-1 by six points at 16 px. Comparing a
sharp query against a blurred candidate is harder than comparing two images
blurred to the same degree, because the mismatch itself costs accuracy.

This turns that observation into a method. Instead of one query encoding used
against every candidate, the query is encoded at each resolution level, and
each gallery crop is compared against the query rendered at ITS OWN
resolution.

  # stage 1, needs the GPU, runs once and caches
  python phase7_resmatch.py --root .\\dataset --encode

  # stage 2, evaluation, no GPU
  python phase7_resmatch.py --root .\\dataset

Deployability
-------------
A gallery crop's resolution is its bounding-box height in pixels, which is
known at search time. This is therefore a real method, not an oracle: nothing
here uses the answer.

Design decision worth stating
-----------------------------
Scores from different query encodings are not automatically on the same
scale, so each level's score vector is standardised BEFORE the per-crop
selection. Gathering raw similarities from different encodings would mix
incomparable numbers and corrupt the ranking - the same failure that sank the
per-crop weighting in Phase 5.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

LEVELS = ["native", "96px", "64px", "48px", "32px", "24px", "16px"]
LEVEL_PX = {"native": 10 ** 6, "96px": 96, "64px": 64, "48px": 48,
            "32px": 32, "24px": 24, "16px": 16}
# a crop of height h is compared against the query rendered at the nearest
# level at or below h
BOUNDS = [20, 28, 40, 56, 80, 128]
ORDER = ["16px", "24px", "32px", "48px", "64px", "96px", "native"]

ALPHA = 0.45


def zrow(v):
    return (v - v.mean()) / (v.std() + 1e-8)


def ap_of(scores, rel):
    n = int(rel.sum())
    if n == 0:
        return np.nan
    o = np.argsort(-scores, kind="stable")
    ro = rel[o]
    pos = np.flatnonzero(ro)
    return float(np.mean((np.arange(n) + 1) / (pos + 1)))


def ci(v, reps=4000, seed=0):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), size=(reps, len(v)))].mean(axis=1)
    return v.mean(), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ------------------------------------------------------------ stage 1
def encode_queries(a_):
    import cv2
    import torch
    from phase3_encode import load_osnet, prep_osnet, encode_osnet
    from phase3_irra import IRRAEncoder
    from phase4_resolution import degrade

    root = Path(a_.root).resolve()
    prw = root / "PRW"
    cache = root.parent / "cache"
    models = root.parent / "models"
    out = cache / "query_degraded.npz"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: no GPU, this will be slow")

    osnet = load_osnet(models, device)
    irra = IRRAEncoder(a_.repo, a_.ckpt, device)

    files = sorted((prw / "query_box").glob("*.jpg"))
    print(f"\nEncoding {len(files)} query boxes at {len(LEVELS)} levels "
          f"({len(files) * len(LEVELS)} encodes)")

    buf = {lv: {"os": [], "ir": []} for lv in LEVELS}
    feats = {lv: {"os": [], "ir": []} for lv in LEVELS}
    ids, cams, frames = [], [], []
    t0 = time.time()

    def flush(lv):
        b = buf[lv]
        if not b["os"]:
            return
        feats[lv]["os"].append(encode_osnet(osnet, b["os"], device, 128))
        feats[lv]["ir"].append(irra.encode_images(b["ir"], 32))
        b["os"].clear(); b["ir"].clear()

    for i, p in enumerate(files, 1):
        img = cv2.imread(str(p))
        if img is None:
            continue
        pid, _, rest = p.stem.partition("_")
        ids.append(int(pid)); frames.append(rest + ".jpg")
        cams.append(int(rest.split("s")[0][1:]))
        for lv in LEVELS:
            d = degrade(img, 0 if lv == "native" else LEVEL_PX[lv])
            buf[lv]["os"].append(prep_osnet(d))
            buf[lv]["ir"].append(irra.preprocess(d))
        if len(buf["native"]["os"]) >= 256:
            for lv in LEVELS:
                flush(lv)
        if i % 500 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(files)}  {i / el:.1f} q/s  "
                  f"ETA {(len(files) - i) / (i / el) / 60:.1f} min")
    for lv in LEVELS:
        flush(lv)

    save = {"ids": np.array(ids, np.int32), "cams": np.array(cams, np.int32),
            "frames": np.array(frames)}
    for lv in LEVELS:
        save[f"os_{lv}"] = np.concatenate(feats[lv]["os"]).astype(np.float16)
        save[f"ir_{lv}"] = np.concatenate(feats[lv]["ir"]).astype(np.float16)
    np.savez_compressed(out, **save)
    n = len(save["ids"])
    print(f"\nSaved {out}  ({n} queries x {len(LEVELS)} levels, "
          f"{out.stat().st_size / 1e6:.0f} MB)")


# ------------------------------------------------------------ stage 2
def evaluate(a_):
    root = Path(a_.root).resolve()
    cache = root.parent / "cache"
    from phase3_baseline import load_split

    qd_path = cache / "query_degraded.npz"
    if not qd_path.exists():
        sys.exit(f"ERROR: {qd_path} missing. Run with --encode first.")
    QD = np.load(qd_path, allow_pickle=True)

    G = np.load(cache / "gallery.npz", allow_pickle=True)
    C = np.load(cache / "captions.npz", allow_pickle=True)
    deg = cache / "degraded"

    sp = load_split(root / "PRW", "frame_test.mat")
    mask = np.array([f in sp for f in G["frames"]]) if sp else np.ones(len(G["frames"]), bool)
    g_pid, g_cam, g_h = G["ids"][mask], G["cams"][mask], G["height"][mask]
    g_native = G["osnet"][mask].astype(np.float32)
    g_ir_native = G["irra"][mask].astype(np.float32)

    q_pid, q_cam, q_frames = QD["ids"], QD["cams"], QD["frames"]
    key = {(int(p), str(f)): i for i, (p, f) in enumerate(zip(q_pid, q_frames))}
    caps_of = {}
    for i, (p, f) in enumerate(zip(C["ids"], C["frames"])):
        r = key.get((int(p), str(f)))
        if r is not None:
            caps_of.setdefault(r, []).append(i)
    c_ir = C["irra"].astype(np.float32)
    qs = [i for i in range(len(q_pid)) if i in caps_of]
    print(f"Gallery {len(g_pid)} crops | queries {len(qs)}\n")

    q_os_lv = {lv: QD[f"os_{lv}"].astype(np.float32) for lv in LEVELS}

    # ---- gallery configurations
    # Each configuration carries its OWN identity and camera arrays. The
    # degraded features were written in Phase 4 with the test-split filter
    # already applied, so they hold 27,778 rows and must not be filtered
    # again; their row order is defined by degraded/meta.npz rather than by
    # gallery.npz.
    rng = np.random.default_rng(a_.seed)
    configs = {}

    lvl_native = np.array([ORDER[int(np.digitize(h, BOUNDS))] for h in g_h])
    configs["native gallery"] = (g_native, g_ir_native, lvl_native,
                                 g_pid, g_cam)

    if (deg / "meta.npz").exists():
        M = np.load(deg / "meta.npz")
        d_pid, d_cam, d_h = M["ids"], M["cams"], M["height"]
        lf = {lv: np.load(deg / f"gallery_{lv}.npz") for lv in LEVELS}
        n_d = len(d_pid)
        if lf["native"]["osnet"].shape[0] != n_d:
            sys.exit(f"ERROR: degraded features have "
                     f"{lf['native']['osnet'].shape[0]} rows but meta has {n_d}")
        print(f"Degraded set: {n_d} crops "
              f"({'aligned with' if n_d == len(g_pid) else 'DIFFERENT SIZE to'}"
              f" the native gallery)\n")

        pick = rng.integers(0, len(LEVELS), size=n_d)
        gm = np.zeros((n_d, 512), np.float32)
        gmi = np.zeros((n_d, 512), np.float32)
        lvm = np.empty(n_d, object)
        for k, lv in enumerate(LEVELS):
            m = pick == k
            if m.any():
                gm[m] = lf[lv]["osnet"][m].astype(np.float32)
                gmi[m] = lf[lv]["irra"][m].astype(np.float32)
                lvm[m] = lv
        configs["mixed resolution"] = (gm, gmi, lvm.astype(str), d_pid, d_cam)

        for lv in ("48px", "32px", "16px"):
            configs[f"all crops at {lv}"] = (
                lf[lv]["osnet"].astype(np.float32),
                lf[lv]["irra"].astype(np.float32),
                np.full(n_d, lv), d_pid, d_cam)
    else:
        print("(degraded features missing - only the native gallery is tested)\n")

    rows = []
    for name, (g_img, g_txt, g_lvl, cfg_pid, cfg_cam) in configs.items():
        print("=" * 72)
        print(f"GALLERY: {name}")
        print("=" * 72)
        uniq, cnt = np.unique(g_lvl, return_counts=True)
        print("  crop levels: " + ", ".join(
            f"{u} {c / len(g_lvl) * 100:.0f}%" for u, c in zip(uniq, cnt)))

        # Thresholded variant. The unconditional method loses above roughly
        # 24 px because degrading a sharp query discards information the
        # gallery crop has already lost. Below that the domain gap dominates
        # and matching pays. Applying it only to small crops keeps the gain
        # and avoids the cost. The threshold is swept and reported.
        THRESH = [0, 20, 28, 40, 56, 10 ** 6]
        res = {"fixed query": [], "resolution-matched": [],
               "fixed query, fused": [], "matched, fused": []}
        for t in THRESH[1:-1]:
            res[f"matched below {t}px, fused"] = []

        for qi in qs:
            keep = (cfg_cam != q_cam[qi])
            idx = np.flatnonzero(keep)
            rel = (cfg_pid[idx] == q_pid[qi])
            if not rel.any():
                continue
            gi = g_img[idx]
            zt = zrow(c_ir[caps_of[qi][0]] @ g_txt[idx].T)

            # baseline: one query encoding for every candidate
            z_fixed = zrow(q_os_lv["native"][qi] @ gi.T)

            # matched: standardise each level's scores BEFORE selecting, so
            # numbers from different encodings are comparable
            z_match = np.empty(len(idx), np.float32)
            lv_idx = g_lvl[idx]
            for lv in LEVELS:
                m = (lv_idx == lv)
                if not m.any():
                    continue
                z_match[m] = zrow(q_os_lv[lv][qi] @ gi.T)[m]

            res["fixed query"].append(ap_of(z_fixed, rel))
            res["resolution-matched"].append(ap_of(z_match, rel))
            res["fixed query, fused"].append(
                ap_of(a_.alpha * z_fixed + (1 - a_.alpha) * zt, rel))
            res["matched, fused"].append(
                ap_of(a_.alpha * z_match + (1 - a_.alpha) * zt, rel))

            # selective: matched encoding only for crops below the threshold
            lv_px = np.array([LEVEL_PX[l] for l in lv_idx], np.float32)
            for t in THRESH[1:-1]:
                small = lv_px < t
                z_sel = np.where(small, z_match, z_fixed)
                res[f"matched below {t}px, fused"].append(
                    ap_of(a_.alpha * z_sel + (1 - a_.alpha) * zt, rel))

        base_i = np.array(res["fixed query"])
        base_f = np.array(res["fixed query, fused"])
        for k, v in res.items():
            v = np.array(v)
            m, lo, hi = ci(v * 100)
            extra = ""
            if k == "resolution-matched":
                d, dl, dh = ci((v - base_i) * 100)
                sig = "SIGNIFICANT" if (dl > 0 or dh < 0) else "ns"
                extra = f"   {d:+5.2f} [{dl:+.2f}, {dh:+.2f}] {sig}"
            elif k == "matched, fused":
                d, dl, dh = ci((v - base_f) * 100)
                sig = "SIGNIFICANT" if (dl > 0 or dh < 0) else "ns"
                extra = f"   {d:+5.2f} [{dl:+.2f}, {dh:+.2f}] {sig}"
            print(f"  {k:<22}{m:6.2f}  [{lo:5.2f}, {hi:5.2f}]{extra}")
            rows.append({"gallery": name, "method": k, "mAP": round(m, 2),
                         "lo": round(lo, 2), "hi": round(hi, 2)})
        print()

    out = cache / "phase7_resmatch.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")
    print("\nThe method should help most where crops are small, since that is")
    print("where the mismatch between a sharp query and a blurred candidate")
    print("costs the most. On the native gallery most crops are already large,")
    print("so little is expected there.")


def main():
    P = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_argument("--root", default="dataset")
    P.add_argument("--repo", default="IRRA")
    P.add_argument("--ckpt", default="models/irra/best.pth")
    P.add_argument("--alpha", type=float, default=ALPHA)
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--encode", action="store_true",
                   help="stage 1: encode queries at every level (needs GPU)")
    a_ = P.parse_args()
    if a_.encode:
        encode_queries(a_)
    else:
        evaluate(a_)


if __name__ == "__main__":
    main()
