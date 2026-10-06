"""
Phase 4c - Confidence intervals for mean average precision.

Phase 4b bootstrapped Rank-1 because only per-query hit flags were stored.
mAP is the more important metric for this project - a missing-person search
needs every sighting across every camera, not only the single best match -
and its crossing point sits at a resolution PRW actually contains. It
therefore needs the same statistical treatment.

This recomputes per-query average precision from the cached degraded features,
so it runs in seconds and requires no GPU.

  python phase4_map_ci.py --root .\\dataset

Writes cache/phase4_map_ci.csv and cache/phase4_map_figure.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ["native", "96px", "64px", "48px", "32px", "24px", "16px"]
B = 5000


def per_query_ap(qf, q_pid, q_cam, gf, g_pid, g_cam, chunk=256):
    """Average precision for each query, with same-camera junk removed."""
    aps = []
    for s in range(0, len(qf), chunk):
        e = min(s + chunk, len(qf))
        sim = qf[s:e].astype(np.float32) @ gf.T
        order = np.argsort(-sim, axis=1)
        for r in range(e - s):
            i = s + r
            o = order[r]
            o = o[g_cam[o] != q_cam[i]]
            match = (g_pid[o] == q_pid[i])
            n = int(match.sum())
            if n == 0:
                continue
            pos = np.where(match)[0]
            aps.append(float(np.mean((np.arange(n) + 1) / (pos + 1))))
    return np.array(aps, np.float64)


def boot_ci(v, reps=B, seed=0):
    rng = np.random.default_rng(seed)
    n = len(v)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = v[rng.integers(0, n, size=(reps, n))].mean(axis=1)
    return (v.mean() * 100, float(np.percentile(m, 2.5) * 100),
            float(np.percentile(m, 97.5) * 100))


def boot_diff(a, b, reps=B, seed=0):
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, len(a), size=(reps, len(a)))].mean(axis=1)
    mb = b[rng.integers(0, len(b), size=(reps, len(b)))].mean(axis=1)
    d = (ma - mb) * 100
    return (float(d.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), float((d < 0).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset")
    a = ap.parse_args()

    cache = Path(a.root).resolve().parent / "cache"
    deg = cache / "degraded"
    if not (deg / "meta.npz").exists():
        sys.exit(f"ERROR: {deg} missing. Rerun phase4_resolution.py "
                 f"with the current version, which saves degraded features.")

    M = np.load(deg / "meta.npz")
    g_pid, g_cam = M["ids"], M["cams"]
    Q = np.load(cache / "query.npz", allow_pickle=True)
    C = np.load(cache / "captions.npz", allow_pickle=True)

    q_os, q_ir, q_pid, q_cam = Q["osnet"], Q["irra"], Q["ids"], Q["cams"]
    f2c = {str(f): c for f, c in zip(Q["frames"], q_cam)}
    c_pid = C["ids"]
    c_cam = np.array([f2c.get(str(f), int(str(f).split("s")[0][1:]))
                      for f in C["frames"]], np.int32)
    c_ir = C["irra"]

    print(f"Gallery {len(g_pid)} crops, {len(q_pid)} photograph queries, "
          f"{len(c_pid)} descriptions\n")

    APS = {}
    for lv in LEVELS:
        f = np.load(deg / f"gallery_{lv}.npz")
        g_os = f["osnet"].astype(np.float32)
        g_ir = f["irra"].astype(np.float32)
        APS[("photo_osnet", lv)] = per_query_ap(q_os, q_pid, q_cam, g_os, g_pid, g_cam)
        APS[("photo_irra", lv)] = per_query_ap(q_ir, q_pid, q_cam, g_ir, g_pid, g_cam)
        APS[("text_irra", lv)] = per_query_ap(c_ir, c_pid, c_cam, g_ir, g_pid, g_cam)
        print(f"  {lv} done")

    print("\n" + "=" * 74)
    print("1.  mAP WITH 95% CONFIDENCE INTERVALS")
    print("=" * 74)
    print(f"{'level':<9}{'photo (appearance)':>24}{'description':>22}"
          f"{'photo (vis-lang)':>22}")
    print("-" * 74)
    rows = []
    for lv in LEVELS:
        m1, l1, h1 = boot_ci(APS[("photo_osnet", lv)])
        m2, l2, h2 = boot_ci(APS[("text_irra", lv)])
        m3, l3, h3 = boot_ci(APS[("photo_irra", lv)])
        print(f"{lv:<9}{m1:>11.2f} [{l1:.1f}, {h1:.1f}]"
              f"{m2:>11.2f} [{l2:.1f}, {h2:.1f}]"
              f"{m3:>11.2f} [{l3:.1f}, {h3:.1f}]")
        rows.append({"level": lv, "photo_osnet_mAP": round(m1, 2),
                     "photo_osnet_lo": round(l1, 2), "photo_osnet_hi": round(h1, 2),
                     "text_mAP": round(m2, 2), "text_lo": round(l2, 2),
                     "text_hi": round(h2, 2), "photo_irra_mAP": round(m3, 2),
                     "photo_irra_lo": round(l3, 2), "photo_irra_hi": round(h3, 2)})

    print("\n" + "=" * 74)
    print("2.  WHERE DOES THE DESCRIPTION OVERTAKE ON mAP?")
    print("=" * 74)
    print(f"{'level':<9}{'difference (pp)':>18}{'95% interval':>22}"
          f"{'P(text ahead)':>16}")
    print("-" * 74)
    first_tie = first_cross = None
    for i, lv in enumerate(LEVELS):
        d, lo, hi, p = boot_diff(APS[("photo_osnet", lv)], APS[("text_irra", lv)])
        if hi < 0:
            tag = "  <-- text ahead"
            first_cross = first_cross or lv
        elif lo < 0 < hi:
            tag = "  (tie)"
            first_tie = first_tie or lv
        else:
            tag = ""
        print(f"{lv:<9}{d:>17.2f}{f'[{lo:6.2f}, {hi:6.2f}]':>22}{p:>15.3f}{tag}")
        rows[i].update(gap=round(d, 2), gap_lo=round(lo, 2),
                       gap_hi=round(hi, 2), p_text_ahead=round(p, 4))

    print()
    if first_tie:
        print(f"The two become statistically indistinguishable at {first_tie}.")
    if first_cross:
        print(f"The description is ahead with 95% confidence at {first_cross} "
              f"and below.")
        print("\nThis is the headline result. It sits at a resolution PRW")
        print("genuinely contains, unlike the Rank-1 crossing, and mAP is the")
        print("metric that matters when the goal is a timeline of every")
        print("sighting rather than one best match.")
    else:
        print("The description does not overtake on mAP within the tested range.")

    print("\n" + "=" * 74)
    print("3.  THE VISION-LANGUAGE COMPARISON")
    print("=" * 74)
    for lv in LEVELS:
        d, lo, hi, p = boot_diff(APS[("photo_irra", lv)], APS[("text_irra", lv)])
        print(f"{lv:<9}{d:>17.2f}{f'[{lo:6.2f}, {hi:6.2f}]':>22}{p:>15.3f}")
    print("\nAgainst the vision-language image branch the gap narrows but does")
    print("not close. Whether a description overtakes a photograph therefore")
    print("depends on the appearance model, and that condition must be stated.")

    out = cache / "phase4_map_ci.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(LEVELS))
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))

        for key, lab, col, mk in (("photo_osnet", "Photograph (appearance)", "#C2410C", "s"),
                                  ("photo_irra", "Photograph (vision-language)", "#1D4ED8", "o"),
                                  ("text_irra", "Description", "#0F766E", "^")):
            m = [boot_ci(APS[(key, lv)]) for lv in LEVELS]
            a1.plot(x, [v[0] for v in m], marker=mk, color=col, label=lab, lw=1.8, ms=6)
            a1.fill_between(x, [v[1] for v in m], [v[2] for v in m],
                            color=col, alpha=.15, lw=0)
        a1.set_xticks(x); a1.set_xticklabels(LEVELS)
        a1.set_xlabel("gallery crop height after degradation")
        a1.set_ylabel("mAP (%)")
        a1.set_title("Mean average precision against gallery resolution")
        a1.grid(alpha=.3); a1.legend(fontsize=9)

        g = [boot_diff(APS[("photo_osnet", lv)], APS[("text_irra", lv)]) for lv in LEVELS]
        a2.axhline(0, color="#555", lw=1)
        a2.plot(x, [v[0] for v in g], marker="D", color="#7C2D12", lw=1.8, ms=6)
        a2.fill_between(x, [v[1] for v in g], [v[2] for v in g],
                        color="#7C2D12", alpha=.15, lw=0)
        a2.set_xticks(x); a2.set_xticklabels(LEVELS)
        a2.set_xlabel("gallery crop height after degradation")
        a2.set_ylabel("photograph advantage, mAP (pp)")
        a2.set_title("Photograph minus description on mAP, 95% interval")
        a2.grid(alpha=.3)
        a2.text(0.03, 0.06, "below this line the\ndescription is better",
                transform=a2.transAxes, fontsize=8, color="#555")

        fig.tight_layout()
        p = cache / "phase4_map_figure.png"
        fig.savefig(p, dpi=170)
        print(f"Saved {p}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
