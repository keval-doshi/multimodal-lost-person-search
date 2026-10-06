"""
Phase 5b - Confidence intervals and power for the fusion result.

Two questions the main run leaves open.

  1. The fusion gain over the photograph alone is large. How large, with an
     interval?
  2. The quality-aware rule gained nothing. Is that a real null, or was the
     study simply unable to detect a gain of that size? A null without a
     power estimate is not a finding.

Both are answered by bootstrapping over queries, paired where the same query
is scored under two methods, which is the correct test here because the
methods are compared on identical queries.

  python phase5_ci.py --root .\\dataset
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from phase5_fusion import (zscore_rows, ap_and_hit, quality_vector,
                           alpha_from_quality, LEVELS, LEVEL_H)

B = 5000


def per_query_ap(z_img, z_txt, alpha, q_pid, q_cam, g_pid, g_cam, qi):
    out = []
    for i in range(len(z_txt)):
        f = alpha * z_img[qi[i]] + (1.0 - alpha) * z_txt[i]
        ap, _ = ap_and_hit(f, g_pid == q_pid[i], g_cam != q_cam[i])
        out.append(np.nan if ap is None else ap)
    return np.array(out)


def paired_ci(a, b, reps=B, seed=0):
    """Bootstrap the paired difference a - b over queries."""
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a[ok] - b[ok]) * 100
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(reps, len(d)))].mean(axis=1)
    return (float(d.mean()), float(np.percentile(m, 2.5)),
            float(np.percentile(m, 97.5)), int(len(d)))


def main():
    P = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_argument("--root", default="dataset")
    P.add_argument("--seed", type=int, default=0)
    a_ = P.parse_args()

    cache = Path(a_.root).resolve().parent / "cache"
    deg = cache / "degraded"
    M = np.load(deg / "meta.npz")
    g_pid, g_cam, g_h = M["ids"], M["cams"], M["height"]
    Q = np.load(cache / "query.npz", allow_pickle=True)
    C = np.load(cache / "captions.npz", allow_pickle=True)

    key = {(int(p), str(f)): i for i, (p, f) in enumerate(zip(Q["ids"], Q["frames"]))}
    qi, keep = [], []
    for i, (p, f) in enumerate(zip(C["ids"], C["frames"])):
        r = key.get((int(p), str(f)))
        if r is not None:
            qi.append(r); keep.append(i)
    qi = np.array(qi); keep = np.array(keep)
    c_ir, c_pid = C["irra"][keep], C["ids"][keep]
    c_cam = Q["cams"][qi]

    # Reproduce the main run's split and gallery exactly: same seed, same
    # sequence of draws, so the held-out queries and the level assignment
    # match what was reported.
    rng = np.random.default_rng(a_.seed)
    ids = np.unique(c_pid)
    rng.shuffle(ids)
    val_ids = set(ids[:len(ids) // 2].tolist())
    is_val = np.array([int(p) in val_ids for p in c_pid])
    vidx = np.flatnonzero(is_val)
    if len(vidx) > 800:
        _ = np.sort(rng.choice(vidx, 800, replace=False))
    tidx = np.flatnonzero(~is_val)
    lvl_idx = rng.integers(0, len(LEVELS), size=len(g_h))

    eff_h = np.array([min(h, LEVEL_H[LEVELS[k]]) for h, k in zip(g_h, lvl_idx)],
                     np.float32)
    qv = quality_vector(eff_h)
    lvl_feats = {lv: np.load(deg / f"gallery_{lv}.npz") for lv in LEVELS}

    # parameters found by the main run
    FIT = {"osnet": {"tuned": 0.45, "centre": 0.45, "slope": 0.0},
           "irra": {"tuned": 0.80, "centre": 0.80, "slope": -0.20}}

    rows = []
    for br in ("osnet", "irra"):
        print("\n" + "=" * 70)
        print(f"IMAGE BRANCH: {br}")
        print("=" * 70)

        gi = np.zeros((len(g_h), 512), np.float32)
        gt = np.zeros((len(g_h), 512), np.float32)
        for k, lv in enumerate(LEVELS):
            m = lvl_idx == k
            gi[m] = lvl_feats[lv][br][m].astype(np.float32)
            gt[m] = lvl_feats[lv]["irra"][m].astype(np.float32)

        q_img = (Q["osnet"] if br == "osnet" else Q["irra"]).astype(np.float32)
        z_img = zscore_rows(q_img @ gi.T)
        z_txt = zscore_rows(c_ir.astype(np.float32) @ gt.T)

        f = FIT[br]
        al_q = alpha_from_quality(qv, f["centre"], f["slope"]).astype(np.float32)

        def ap_for(alpha):
            return per_query_ap(z_img, z_txt[tidx], alpha, c_pid[tidx],
                                c_cam[tidx], g_pid, g_cam, qi[tidx])

        ap_img = ap_for(1.0)
        ap_txt = ap_for(0.0)
        ap_fix = ap_for(0.5)
        ap_tun = ap_for(f["tuned"])
        ap_qua = ap_for(al_q)

        for lab, v in (("image only", ap_img), ("text only", ap_txt),
                       ("fixed 0.5", ap_fix), ("tuned fixed", ap_tun),
                       ("quality-aware", ap_qua)):
            ok = ~np.isnan(v)
            print(f"  {lab:<15}mAP {v[ok].mean() * 100:6.2f}")

        print("\n  paired differences, 95% bootstrap intervals")
        comps = [("fusion over image alone", ap_tun, ap_img),
                 ("fusion over text alone", ap_tun, ap_txt),
                 ("tuned over fixed 0.5", ap_tun, ap_fix),
                 ("quality-aware over tuned", ap_qua, ap_tun)]
        for lab, x, y in comps:
            d, lo, hi, n = paired_ci(x, y)
            sig = "significant" if (lo > 0 or hi < 0) else "not significant"
            print(f"    {lab:<26}{d:+7.2f}  [{lo:+6.2f}, {hi:+6.2f}]  {sig}")
            rows.append({"branch": br, "comparison": lab, "diff_mAP": round(d, 3),
                         "lo": round(lo, 3), "hi": round(hi, 3),
                         "n_queries": n, "significant": sig == "significant"})

        # ---- power: what gain could this study have detected?
        d, lo, hi, n = paired_ci(ap_qua, ap_tun)
        half = (hi - lo) / 2
        print(f"\n  POWER for the quality-aware comparison")
        print(f"    observed difference {d:+.3f} mAP over {n} queries")
        print(f"    interval half-width {half:.3f} mAP")
        print(f"    A true gain above about {half:.2f} mAP would have been")
        print(f"    detected. The null therefore rules out gains larger than")
        print(f"    roughly {half:.2f} mAP, and does not rule out a smaller one.")
        rows.append({"branch": br, "comparison": "detectable threshold",
                     "diff_mAP": round(half, 3), "lo": None, "hi": None,
                     "n_queries": n, "significant": None})

    out = cache / "phase5_ci.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
