"""
Phase 4b - Analysis of the resolution study.

Turns the raw curves into finished results: confidence intervals on every
figure, a significance test on the crossing point, decay rates, and the
publication figure.

  python phase4_analysis.py --root .\\dataset

Reads cache/phase4_resolution.csv and cache/phase4_hits.npz.
Writes cache/phase4_analysis.csv and cache/phase4_figure.png

Why confidence intervals matter here
------------------------------------
Rank-1 is a proportion over roughly 2,000 queries, so differences of a point
or two are not distinguishable from sampling noise. The crossing point is the
central claim of this phase, and claiming it without an interval invites the
obvious question of whether it is real. Bootstrapping over queries answers
that directly.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ["native", "96px", "64px", "48px", "32px", "24px", "16px"]
HEIGHTS = [0, 96, 64, 48, 32, 24, 16]
B = 5000                                    # bootstrap resamples


def boot_ci(hits, reps=B, seed=0, alpha=0.05):
    """Percentile bootstrap CI for the mean of a boolean vector."""
    rng = np.random.default_rng(seed)
    n = len(hits)
    if n == 0:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, n, size=(reps, n))
    means = hits[idx].mean(axis=1)
    return (hits.mean() * 100,
            float(np.percentile(means, alpha / 2 * 100) * 100),
            float(np.percentile(means, (1 - alpha / 2) * 100) * 100))


def boot_diff(a, b, reps=B, seed=0):
    """
    CI for the difference of two independent proportions, and the share of
    resamples where a exceeds b. The two query sets differ in size (photograph
    queries against description queries), so they are resampled separately.
    """
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    ma = a[rng.integers(0, na, size=(reps, na))].mean(axis=1)
    mb = b[rng.integers(0, nb, size=(reps, nb))].mean(axis=1)
    d = (ma - mb) * 100
    return (float(d.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), float((d < 0).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset")
    a = ap.parse_args()

    cache = Path(a.root).resolve().parent / "cache"
    csv_p = cache / "phase4_resolution.csv"
    hits_p = cache / "phase4_hits.npz"
    if not csv_p.exists():
        sys.exit(f"ERROR: {csv_p} missing. Run phase4_resolution.py first.")
    df = pd.read_csv(csv_p)

    if not hits_p.exists():
        print("WARNING: phase4_hits.npz missing - rerun phase4_resolution.py")
        print("with the current version to enable confidence intervals.\n")
        H = None
    else:
        H = np.load(hits_p)

    # ---------------------------------------------------------- 1
    print("=" * 72)
    print("1.  RANK-1 WITH 95% CONFIDENCE INTERVALS")
    print("=" * 72)
    rows = []
    if H is not None:
        print(f"{'level':<9}{'photo (appearance)':>26}{'description':>24}")
        print("-" * 72)
        for lv in LEVELS:
            po = H[f"photo_osnet_{lv}"]
            tx = H[f"text_irra_{lv}"]
            m1, lo1, hi1 = boot_ci(po)
            m2, lo2, hi2 = boot_ci(tx)
            print(f"{lv:<9}{m1:>13.2f} [{lo1:.1f}, {hi1:.1f}]"
                  f"{m2:>13.2f} [{lo2:.1f}, {hi2:.1f}]")
            rows.append({"level": lv, "photo_osnet": m1,
                         "photo_osnet_lo": lo1, "photo_osnet_hi": hi1,
                         "text": m2, "text_lo": lo2, "text_hi": hi2,
                         "n_photo": len(po), "n_text": len(tx)})
        print(f"\n{len(H['photo_osnet_native'])} photograph queries, "
              f"{len(H['text_irra_native'])} descriptions, "
              f"{B} bootstrap resamples")

    # ---------------------------------------------------------- 2
    print("\n" + "=" * 72)
    print("2.  IS THE CROSSING REAL?")
    print("=" * 72)
    if H is not None:
        print("Photograph minus description, with a 95% interval and the")
        print("probability that the description is genuinely ahead.\n")
        print(f"{'level':<9}{'difference (pp)':>20}{'95% interval':>22}{'P(text ahead)':>16}")
        print("-" * 72)
        cross = None
        for lv in LEVELS:
            d, lo, hi, p = boot_diff(H[f"photo_osnet_{lv}"], H[f"text_irra_{lv}"])
            flag = "  <-- crosses" if hi < 0 else ("  (overlaps 0)" if lo < 0 < hi else "")
            print(f"{lv:<9}{d:>19.2f}{f'[{lo:6.2f}, {hi:6.2f}]':>22}"
                  f"{p:>15.3f}{flag}")
            if cross is None and hi < 0:
                cross = lv
            for r in rows:
                if r["level"] == lv:
                    r.update(gap=d, gap_lo=lo, gap_hi=hi, p_text_ahead=p)

        print()
        if cross:
            print(f"The description is ahead with 95% confidence at {cross}")
            print("and below. The crossing is not a sampling artefact.")
        else:
            print("No level shows the description ahead with 95% confidence.")
            print("Report the narrowing gap rather than a crossing point.")

    # ---------------------------------------------------------- 3
    print("\n" + "=" * 72)
    print("3.  HOW FAST EACH MODALITY DECAYS")
    print("=" * 72)
    piv = df[df["method"].isin(["photo_osnet", "photo_irra", "text_irra"])] \
        .pivot(index="target_height", columns="method", values="Rank-1") \
        .reindex(HEIGHTS)
    print(f"{'method':<24}{'native':>10}{'16px':>9}{'retained':>11}")
    print("-" * 72)
    for col, label in (("photo_osnet", "Photograph (appearance)"),
                       ("photo_irra", "Photograph (vision-lang)"),
                       ("text_irra", "Description")):
        nat, low = piv[col][0], piv[col][16]
        print(f"{label:<24}{nat:>9.2f}%{low:>8.2f}%{low / nat * 100:>10.1f}%")
    print("\nThe description retains a far larger share of its accuracy.")
    print("That difference in decay rate, not the crossing point alone, is")
    print("what a quality-aware fusion rule exploits.")

    # ---------------------------------------------------------- 4
    print("\n" + "=" * 72)
    print("4.  mAP, WHICH REWARDS FINDING EVERY SIGHTING")
    print("=" * 72)
    pm = df[df["method"].isin(["photo_osnet", "photo_irra", "text_irra"])] \
        .pivot(index="target_height", columns="method", values="mAP") \
        .reindex(HEIGHTS)
    print(pm.round(2).to_string())
    gap_r1 = piv["photo_osnet"] - piv["text_irra"]
    gap_map = pm["photo_osnet"] - pm["text_irra"]
    print(f"\nGap at native:  Rank-1 {gap_r1[0]:+.2f} pp, mAP {gap_map[0]:+.2f} pp")
    print(f"Gap at 16px:    Rank-1 {gap_r1[16]:+.2f} pp, mAP {gap_map[16]:+.2f} pp")
    print("\nFor a missing-person search mAP matters more than Rank-1, because")
    print("the operator wants every sighting across every camera, not just the")
    print("single best match.")

    # ---------------------------------------------------------- 5
    if "photo_irra_matched" in set(df["method"]):
        print("\n" + "=" * 72)
        print("5.  DIAGNOSTIC: THE VISION-LANGUAGE ANOMALY")
        print("=" * 72)
        pmm = df[df["method"].isin(["photo_irra", "photo_irra_matched"])] \
            .pivot(index="target_height", columns="method", values="Rank-1") \
            .reindex(HEIGHTS)
        print(f"{'level':<9}{'gallery only':>15}{'both degraded':>17}")
        print("-" * 72)
        for h, lv in zip(HEIGHTS, LEVELS):
            print(f"{lv:<9}{pmm['photo_irra'][h]:>14.2f}%"
                  f"{pmm['photo_irra_matched'][h]:>16.2f}%")
        b1 = pmm["photo_irra"][96] - pmm["photo_irra"][0]
        b2 = pmm["photo_irra_matched"][96] - pmm["photo_irra_matched"][0]
        print(f"\nChange from native to 96px:")
        print(f"  gallery degraded only   {b1:+.2f} pp")
        print(f"  both sides degraded     {b2:+.2f} pp")
        if b1 > 0.5 and b2 < b1 / 2:
            print("\nThe improvement largely disappears once both sides are")
            print("degraded, which is what a domain-gap explanation predicts:")
            print("mild blurring moves the sharp PRW crops toward the softer")
            print("imagery the model was fine-tuned on. It is not an")
            print("information effect, and should be reported as an artefact.")
        elif b1 > 0.5:
            print("\nThe improvement survives matched degradation, so the")
            print("domain-gap explanation is not supported. Investigate before")
            print("committing to an interpretation in the write-up.")

    if rows:
        out = cache / "phase4_analysis.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nSaved {out}")

    # ---------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(len(LEVELS))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

        series = (("photo_osnet", "Photograph (appearance)", "#C2410C", "s"),
                  ("photo_irra", "Photograph (vision-language)", "#1D4ED8", "o"),
                  ("text_irra", "Description", "#0F766E", "^"))

        for col, lab, c, mk in series:
            y = [piv[col][h] for h in HEIGHTS]
            ax1.plot(x, y, marker=mk, color=c, label=lab, lw=1.8, ms=6)
            if H is not None:
                key = {"photo_osnet": "photo_osnet", "photo_irra": "photo_irra",
                       "text_irra": "text_irra"}[col]
                lo, hi = [], []
                for lv in LEVELS:
                    _, l, h_ = boot_ci(H[f"{key}_{lv}"])
                    lo.append(l); hi.append(h_)
                ax1.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0)

        ax1.set_xticks(x); ax1.set_xticklabels(LEVELS)
        ax1.set_xlabel("gallery crop height after degradation")
        ax1.set_ylabel("Rank-1 (%)")
        ax1.set_title("Retrieval accuracy against gallery resolution")
        ax1.grid(alpha=.3); ax1.legend(fontsize=9)

        gap = [piv["photo_osnet"][h] - piv["text_irra"][h] for h in HEIGHTS]
        ax2.axhline(0, color="#555", lw=1)
        ax2.plot(x, gap, marker="D", color="#7C2D12", lw=1.8, ms=6)
        if H is not None:
            lo, hi = [], []
            for lv in LEVELS:
                _, l, h_, _ = boot_diff(H[f"photo_osnet_{lv}"], H[f"text_irra_{lv}"])
                lo.append(l); hi.append(h_)
            ax2.fill_between(x, lo, hi, color="#7C2D12", alpha=0.15, lw=0)
        ax2.set_xticks(x); ax2.set_xticklabels(LEVELS)
        ax2.set_xlabel("gallery crop height after degradation")
        ax2.set_ylabel("photograph advantage (pp)")
        ax2.set_title("Photograph minus description, 95% interval")
        ax2.grid(alpha=.3)
        ax2.text(0.03, 0.06, "below this line the\ndescription is better",
                 transform=ax2.transAxes, fontsize=8, color="#555")

        fig.tight_layout()
        p = cache / "phase4_figure.png"
        fig.savefig(p, dpi=170)
        print(f"Saved {p}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
