"""
Phase 5 (revised) - Quality-aware fusion.

Supersedes the first attempt, which had two flaws:

  1. It used the vision-language image encoder. Phase 4 established that with
     that encoder the photograph leads the description at every resolution -
     the gap narrows from +28.6 to +15.7 points but never closes. Testing a
     quality-aware rule there asks it to exploit a crossing that does not
     exist. The crossing is in the APPEARANCE branch, so that is the branch
     to fuse.

  2. Its "mixed" gallery assigned resolution levels from native crop height.
     Since PRW's median height is 150 px, 60 per cent of crops landed at
     native and only 5.7 per cent below 48 px, so the mixed setting was
     effectively the native setting and quality barely varied.

  python phase5_fusion.py --root .\\dataset --branch both

Diagnostic
----------
The decisive question is whether the best constant alpha actually changes
with crop quality. If it is flat, no quality-dependent rule can help and the
negative result has a clear cause. Section 2 of the output measures this
directly by finding the best constant within each quality band.

Note on the quality signal
--------------------------
Quality is taken as the effective height after degradation, which is known
exactly because the degradation was applied here. A deployed system would
have to estimate it from the crop, for example by measuring sharpness. This
is therefore an oracle signal and an upper bound: if a quality-aware rule
gains nothing even with perfect quality information, it will gain nothing
with an estimate either. That makes a negative result stronger, and a
positive one provisional pending a measurable estimator.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ["native", "96px", "64px", "48px", "32px", "24px", "16px"]
LEVEL_H = {"native": 256, "96px": 96, "64px": 64, "48px": 48,
           "32px": 32, "24px": 24, "16px": 16}


def zscore_rows(S):
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True) + 1e-8
    return (S - mu) / sd


def ap_and_hit(scores, rel, valid):
    s, r = scores[valid], rel[valid]
    n = int(r.sum())
    if n == 0:
        return None, None
    o = np.argsort(-s, kind="stable")
    ro = r[o]
    pos = np.flatnonzero(ro)
    return float(np.mean((np.arange(n) + 1) / (pos + 1))), bool(ro[0])


def evaluate(z_img, z_txt, alpha, q_pid, q_cam, g_pid, g_cam, qi):
    aps, hits = [], []
    for i in range(len(z_txt)):
        f = alpha * z_img[qi[i]] + (1.0 - alpha) * z_txt[i]
        ap, h = ap_and_hit(f, g_pid == q_pid[i], g_cam != q_cam[i])
        if ap is not None:
            aps.append(ap); hits.append(h)
    if not aps:
        return 0.0, 0.0
    return float(np.mean(aps) * 100), float(np.mean(hits) * 100)


def quality_vector(eff_h, lo=16.0, hi=160.0):
    return np.clip((np.asarray(eff_h, np.float32) - lo) / (hi - lo), 0.0, 1.0)


def alpha_from_quality(q, centre, slope):
    """
    Linear in quality, centred so that slope = 0 reproduces a constant exactly.

    The model class MUST nest the tuned-constant baseline. With a sigmoid
    parameterised over a coarse grid the reachable constants were only
    {0.02, 0.05, 0.12, 0.27, 0.50, 0.73, 0.88}, so a tuned constant of 0.45
    could not be expressed and the rule was compared against a baseline
    outside its own search space. Any apparent shortfall then measures the
    parameterisation rather than the idea.

    Here alpha = centre + slope * (q - 0.5), clipped to [0, 1]. Setting
    slope = 0 gives alpha = centre, so the search space strictly contains
    every constant on the same grid.
    """
    return np.clip(centre + slope * (np.asarray(q, np.float32) - 0.5), 0.0, 1.0)


def main():
    P = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_argument("--root", default="dataset")
    P.add_argument("--branch", default="both", choices=["osnet", "irra", "both"])
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--max-val", type=int, default=800)
    a_ = P.parse_args()

    cache = Path(a_.root).resolve().parent / "cache"
    deg = cache / "degraded"
    if not (deg / "meta.npz").exists():
        sys.exit(f"ERROR: {deg} missing. Run phase4_resolution.py first.")

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
    c_ir = C["irra"][keep]
    c_pid = C["ids"][keep]
    c_cam = Q["cams"][qi]

    ids = np.unique(c_pid)
    rng = np.random.default_rng(a_.seed)
    rng.shuffle(ids)
    val_ids = set(ids[:len(ids) // 2].tolist())
    is_val = np.array([int(p) in val_ids for p in c_pid])
    vidx = np.flatnonzero(is_val)
    if len(vidx) > a_.max_val:
        vidx = np.sort(rng.choice(vidx, a_.max_val, replace=False))
    tidx = np.flatnonzero(~is_val)
    print(f"Identity-disjoint split: {len(val_ids)} validation / "
          f"{len(ids) - len(val_ids)} test identities")
    print(f"Queries: {len(vidx)} used for fitting, {len(tidx)} held out")

    # ---- genuinely mixed gallery: level assigned at random, so quality
    # varies independently of how large the person happened to be
    lvl_idx = rng.integers(0, len(LEVELS), size=len(g_h))
    eff_h = np.array([min(h, LEVEL_H[LEVELS[k]]) for h, k in zip(g_h, lvl_idx)],
                     np.float32)
    print("\nMixed gallery composition (levels assigned at random):")
    for k, lv in enumerate(LEVELS):
        n = int((lvl_idx == k).sum())
        print(f"  {lv:<8}{n:>7}  ({n / len(g_h) * 100:5.1f}%)")
    print(f"effective height  median {np.median(eff_h):.0f}px  "
          f"IQR {np.percentile(eff_h, 25):.0f}-{np.percentile(eff_h, 75):.0f}  "
          f"std {eff_h.std():.1f}")

    lvl_feats = {lv: np.load(deg / f"gallery_{lv}.npz") for lv in LEVELS}

    branches = ["osnet", "irra"] if a_.branch == "both" else [a_.branch]
    rows = []

    for br in branches:
        print("\n" + "=" * 72)
        print(f"IMAGE BRANCH: {br}"
              f"{'   (where the Phase 4 crossing lives)' if br == 'osnet' else ''}")
        print("=" * 72)

        gal_img = np.zeros((len(g_h), 512), np.float32)
        gal_txt = np.zeros((len(g_h), 512), np.float32)
        for k, lv in enumerate(LEVELS):
            m = lvl_idx == k
            if m.any():
                gal_img[m] = lvl_feats[lv][br][m].astype(np.float32)
                gal_txt[m] = lvl_feats[lv]["irra"][m].astype(np.float32)

        q_img = (Q["osnet"] if br == "osnet" else Q["irra"]).astype(np.float32)
        z_img = zscore_rows(q_img @ gal_img.T)
        z_txt = zscore_rows(c_ir.astype(np.float32) @ gal_txt.T)
        qv = quality_vector(eff_h)

        def run(idx, alpha):
            return evaluate(z_img, z_txt[idx], alpha, c_pid[idx], c_cam[idx],
                            g_pid, g_cam, qi[idx])

        # ---------------- 1. tune the constant
        best_a, best_m = 0.5, -1.0
        for al in np.round(np.arange(0.0, 1.01, 0.05), 2):
            m, _ = run(vidx, float(al))
            if m > best_m:
                best_a, best_m = float(al), m
        print(f"\n1.  Best constant alpha on validation: {best_a:.2f}  "
              f"(mAP {best_m:.2f})")

        # ---------------- 2. does the optimum move with quality?
        print("\n2.  DIAGNOSTIC - best constant alpha within each quality band")
        print("    A flat column means no quality-dependent rule can help.")
        print(f"    {'band':<14}{'crops':>8}{'best alpha':>13}{'mAP':>9}")
        print("    " + "-" * 46)
        band_best = []
        for lo, hi in [(0, 24), (24, 40), (40, 72), (72, 128), (128, 10 ** 6)]:
            sel = (eff_h >= lo) & (eff_h < hi)
            if sel.sum() < 500:
                continue
            zi, zt = z_img[:, sel], z_txt[:, sel]
            gp, gc = g_pid[sel], g_cam[sel]
            ba, bm = 0.5, -1.0
            for al in np.round(np.arange(0.0, 1.01, 0.1), 2):
                m, _ = evaluate(zi, zt[vidx], float(al), c_pid[vidx],
                                c_cam[vidx], gp, gc, qi[vidx])
                if m > bm:
                    ba, bm = float(al), m
            lab = f"{lo}-{hi if hi < 10 ** 6 else '+'}px"
            print(f"    {lab:<14}{int(sel.sum()):>8}{ba:>13.2f}{bm:>9.2f}")
            band_best.append(ba)
            rows.append({"branch": br, "setting": "band", "method": lab,
                         "alpha": ba, "mAP": round(bm, 2), "Rank-1": None})

        if len(band_best) >= 2:
            spread = max(band_best) - min(band_best)
            print(f"\n    spread in optimal alpha across bands: {spread:.2f}")
            if spread < 0.1:
                print("    -> essentially flat. The best balance does not depend")
                print("       on crop quality, so a quality-aware rule has")
                print("       nothing to exploit. This explains section 3.")
            else:
                print("    -> the optimum shifts with quality, so a quality-aware")
                print("       rule has something real to exploit.")

        # ---------------- 3. fit and evaluate
        # The centre grid matches the constant sweep exactly, so slope = 0
        # reproduces the tuned constant and the search cannot do worse.
        best_ab, best_qm = (best_a, 0.0), -1.0
        for centre in np.round(np.arange(0.0, 1.01, 0.05), 2):
            for slope in np.round(np.arange(-1.2, 1.21, 0.2), 2):
                al = alpha_from_quality(qv, float(centre), float(slope)).astype(np.float32)
                m, _ = run(vidx, al)
                if m > best_qm:
                    best_ab, best_qm = (float(centre), float(slope)), m
        cc, ss = best_ab
        al_q = alpha_from_quality(qv, cc, ss).astype(np.float32)
        print(f"\n3.  Quality-aware rule fitted: centre={cc:.2f}, slope={ss:.2f} "
              f"(validation mAP {best_qm:.2f})")
        print(f"    alpha spans {al_q.min():.2f} to {al_q.max():.2f} "
              f"across gallery crops")
        if abs(ss) < 1e-9:
            print("    slope is zero: the fit collapsed to a constant, so the")
            print("    quality signal added nothing on the validation split.")
        else:
            print(f"    a positive slope means poor crops lean on the text and")
            print(f"    good crops lean on the photograph")
        print(f"    sanity: quality-aware validation mAP {best_qm:.2f} must be "
              f">= tuned constant {best_m:.2f}")
        assert best_qm >= best_m - 1e-6, \
            "the search space must contain the tuned constant"

        print("\n    Held-out test half:")
        res = {}
        for lab, al in (("image only", 1.0), ("text only", 0.0),
                        ("fixed 0.5", 0.5), ("tuned fixed", best_a),
                        ("quality-aware", al_q)):
            m, r1 = run(tidx, al)
            res[lab] = m
            tag = f"  (alpha {al:.2f})" if isinstance(al, float) else ""
            print(f"      {lab:<15}mAP {m:6.2f}   Rank-1 {r1:6.2f}{tag}")
            rows.append({"branch": br, "setting": "mixed", "method": lab,
                         "alpha": al if isinstance(al, float) else None,
                         "qa_centre": cc, "qa_slope": ss,
                         "mAP": round(m, 2), "Rank-1": round(r1, 2)})

        print(f"\n    fusion gain over image alone:   "
              f"{res['tuned fixed'] - res['image only']:+.2f} mAP")
        print(f"    quality-aware over tuned fixed: "
              f"{res['quality-aware'] - res['tuned fixed']:+.2f} mAP")

    df = pd.DataFrame(rows)
    out = cache / "phase5_fusion_v2.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")

    if len(branches) > 1:
        print("\n" + "=" * 72)
        print("SUMMARY - mAP on the held-out half, mixed-resolution gallery")
        print("=" * 72)
        print(df[df["setting"] == "mixed"]
              .pivot(index="method", columns="branch", values="mAP")
              .reindex(["image only", "text only", "fixed 0.5",
                        "tuned fixed", "quality-aware"]).to_string())


if __name__ == "__main__":
    main()
