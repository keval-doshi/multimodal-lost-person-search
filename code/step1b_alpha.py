"""
Step 1b - Does the description help, once the balance is tuned fairly?

The corrected evaluation found that adding the description helps OSNet by
about ten mAP points but adds nothing significant to IRRA. The balance
parameter alpha, however, was tuned in Phase 5 on a different setup (one
photograph, one description, old protocol). Before concluding anything, this
script re-tunes alpha for the CURRENT configuration (two photographs, two
descriptions, standard protocol) on the VALIDATION identities only, and then
tests on the held-out identities.

It does this for both image encoders and for six galleries - native, mixed,
and every crop degraded to 48, 32, 24 and 16 pixels - so it also answers
RQ3 at system level: does the description start to help the strong encoder
once people are small enough?

  python step1b_alpha.py --root .\\dataset --limit 60     # smoke test
  python step1b_alpha.py --root .\\dataset                # full run, ~15 min

Reads the same caches as step1_corrected_eval.py and reuses its tested
functions. Writes cache/step1/alpha_curves.csv, alpha_results.csv and
alpha_figure.png.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from step1_corrected_eval import zrow, unit, rerank, metrics, cboot, LEVELS

GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def ap_e2e(scores, rel, n_gt):
    r = metrics(scores, rel, n_gt)
    return np.nan if r is None else r[1]


def run_gallery(gname, G_img, G_ir, GP, GC, queries, grid=GRID):
    """
    queries: list of dicts with pid, excl, n_gt, and the query vectors
      'img' (dict branch -> 2-photo vector), 'txt' (averaged description vector)
    G_img: dict branch -> gallery feature matrix for that branch
    Returns per-query curve rows and a function to score one alpha.
    """
    rows = []
    for q in queries:
        keep = ~((GP == q["pid"]) & np.isin(GC, q["excl"]))
        idx = np.flatnonzero(keep)
        rel = GP[idx] == q["pid"]
        zt = zrow(q["txt"] @ G_ir[idx].T)
        for br, G in G_img.items():
            zi = zrow(q["img"][br] @ G[idx].T)
            for al in grid:
                rows.append({"gallery": gname, "branch": br, "alpha": float(al),
                             "qi": q["qi"], "pid": q["pid"], "split": q["split"],
                             "ap": ap_e2e(al * zi + (1 - al) * zt, rel, q["n_gt"])})
    return rows


def heldout_eval(gname, G_img, G_ir, GP, GC, queries, alphas):
    """Image-only vs tuned fusion, each with and without re-ranking."""
    rows = []
    for q in queries:
        keep = ~((GP == q["pid"]) & np.isin(GC, q["excl"]))
        idx = np.flatnonzero(keep)
        rel = GP[idx] == q["pid"]
        zt = zrow(q["txt"] @ G_ir[idx].T)
        for br, G in G_img.items():
            gi = G[idx]
            zi = zrow(q["img"][br] @ gi.T)
            al = alphas[(gname, br)]
            fused = al * zi + (1 - al) * zt
            for meth, s in (("image only", zi), ("fusion", fused),
                            ("image only + re-ranking", rerank(zi, gi)),
                            ("fusion + re-ranking", rerank(fused, gi))):
                r = metrics(s, rel, q["n_gt"])
                rows.append({"gallery": gname, "branch": br, "method": meth,
                             "alpha": al, "qi": q["qi"], "pid": q["pid"],
                             "ap_e2e": r[1], "ap_cond": r[0], "r1_e2e": r[3]})
    return rows


def main():
    P = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_argument("--root", default="dataset")
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--limit", type=int, default=0)
    a = P.parse_args()

    from phase3_baseline import load_split
    from phase3_encode import load_gt_boxes

    t0 = time.time()
    root = Path(a.root).resolve(); prw = root / "PRW"
    cache = root.parent / "cache"; out = cache / "step1"; out.mkdir(exist_ok=True)

    G = np.load(cache / "gallery.npz", allow_pickle=True)
    Q = np.load(cache / "query.npz", allow_pickle=True)
    C = np.load(cache / "captions.npz", allow_pickle=True)
    tf = load_split(prw, "frame_test.mat")
    mask = np.array([f in tf for f in G["frames"]])
    GP, GC = G["ids"][mask], G["cams"][mask]

    gt = load_gt_boxes(prw, sorted(tf))
    gtc = {}
    for f, (_, ids) in gt.items():
        cam = int(f.split("s")[0][1:])
        for p in ids:
            if p > 0: gtc[(int(p), cam)] = gtc.get((int(p), cam), 0) + 1

    q_os, q_ir = Q["osnet"].astype(np.float32), Q["irra"].astype(np.float32)
    q_pid, q_cam, q_frames = Q["ids"], Q["cams"], Q["frames"]
    c_ir = C["irra"].astype(np.float32)
    key = {(int(p), str(f)): i for i, (p, f) in enumerate(zip(q_pid, q_frames))}
    caps = {}
    for i, (p, f) in enumerate(zip(C["ids"], C["frames"])):
        r = key.get((int(p), str(f)))
        if r is not None: caps.setdefault(r, []).append(i)
    by_id = {}
    for i in range(len(q_pid)):
        if i in caps: by_id.setdefault(int(q_pid[i]), []).append(i)
    qs = [i for i in range(len(q_pid)) if i in caps]
    ids = np.unique(q_pid[qs]); sh = ids.copy()
    np.random.default_rng(a.seed).shuffle(sh)
    val_ids = set(sh[:len(sh) // 2].tolist())      # same split as every phase

    queries = []
    for qi in qs:
        pid = int(q_pid[qi])
        grp = [qi] + [j for j in by_id[pid] if j != qi][:1]
        excl = sorted(set(int(c) for c in q_cam[grp]))
        n_gt = sum(v for (p, c), v in gtc.items() if p == pid and c not in excl)
        if n_gt == 0: continue
        cl = caps[qi]
        queries.append({"qi": qi, "pid": pid, "excl": excl, "n_gt": n_gt,
                        "split": "val" if pid in val_ids else "test",
                        "img": {"osnet": unit(q_os[grp].mean(0)), "irra": unit(q_ir[grp].mean(0))},
                        "txt": unit(c_ir[cl[:2]].mean(0))})
    val = [q for q in queries if q["split"] == "val"]
    test = [q for q in queries if q["split"] == "test"]
    if a.limit: val, test = val[:a.limit], test[:a.limit]
    print(f"validation queries {len(val)} | held-out queries {len(test)}")

    # galleries: native, mixed (same random assignment as step 1), and uniform levels
    gals = {"native": ({"osnet": G["osnet"][mask].astype(np.float32),
                        "irra": G["irra"][mask].astype(np.float32)},
                       G["irra"][mask].astype(np.float32))}
    deg = cache / "degraded"
    lf = {lv: np.load(deg / f"gallery_{lv}.npz") for lv in LEVELS}
    Mt = np.load(deg / "meta.npz")
    if not (np.array_equal(Mt["ids"], GP) and np.array_equal(Mt["cams"], GC)):
        sys.exit("ERROR: degraded features are not row-aligned with the gallery.")
    pick = np.random.default_rng(a.seed).integers(0, len(LEVELS), size=len(GP))
    mo = np.zeros_like(gals["native"][1]); mi = np.zeros_like(mo)
    for k, lv in enumerate(LEVELS):
        m = pick == k
        mo[m] = lf[lv]["osnet"][m]; mi[m] = lf[lv]["irra"][m]
    gals["mixed"] = ({"osnet": mo, "irra": mi}, mi)
    for lv in ("48px", "32px", "24px", "16px"):
        o = lf[lv]["osnet"].astype(np.float32); i_ = lf[lv]["irra"].astype(np.float32)
        gals[lv] = ({"osnet": o, "irra": i_}, i_)

    # ---- 1. sweep alpha on VALIDATION identities
    curves = []
    for g, (Gimg, Gir) in gals.items():
        curves += run_gallery(g, Gimg, Gir, GP, GC, val)
        print(f"  swept {g}  [{(time.time() - t0) / 60:.1f} min]")
    cv = pd.DataFrame(curves)
    curve = cv.groupby(["gallery", "branch", "alpha"]).ap.mean().mul(100).reset_index()
    curve.to_csv(out / "alpha_curves.csv", index=False)
    alphas = {}
    for (g, br), s in curve.groupby(["gallery", "branch"]):
        # exact ties resolve toward the photograph, the more conservative choice
        top = s[s.ap >= s.ap.max() - 1e-9]
        alphas[(g, br)] = float(top.alpha.max())

    # ---- 2. evaluate once on HELD-OUT identities
    res = []
    for g, (Gimg, Gir) in gals.items():
        res += heldout_eval(g, Gimg, Gir, GP, GC, test, alphas)
    df = pd.DataFrame(res)
    df.to_csv(out / "alpha_perquery.csv", index=False)

    rows = []
    for (g, br), s in df.groupby(["gallery", "branch"], sort=False):
        def get(m): return s[s.method == m].set_index("qi")
        io, fu, ior, fur = get("image only"), get("fusion"), get("image only + re-ranking"), get("fusion + re-ranking")
        def pdiff(x, y):
            j = x.join(y[["ap_e2e"]], rsuffix="_b")
            return cboot((j.ap_e2e - j.ap_e2e_b).values * 100, j.pid.values)
        d1 = pdiff(fu, io); d2 = pdiff(fur, ior)
        rows.append({"gallery": g, "branch": br, "alpha_tuned": alphas[(g, br)],
                     "image_only": round(io.ap_e2e.mean() * 100, 2),
                     "fusion": round(fu.ap_e2e.mean() * 100, 2),
                     "desc_gain": round(d1[0], 2), "desc_lo": round(d1[1], 2), "desc_hi": round(d1[2], 2),
                     "image_rr": round(ior.ap_e2e.mean() * 100, 2),
                     "fusion_rr": round(fur.ap_e2e.mean() * 100, 2),
                     "desc_gain_rr": round(d2[0], 2), "rr_lo": round(d2[1], 2), "rr_hi": round(d2[2], 2)})
    R = pd.DataFrame(rows)
    R.to_csv(out / "alpha_results.csv", index=False)

    print("\n" + "=" * 100)
    print("DOES THE DESCRIPTION HELP?  alpha tuned on validation identities, tested on held-out identities")
    print("end-to-end mAP; gain = fusion minus image-only, identity-level 95% interval")
    print("=" * 100)
    print(f"{'gallery':<8}{'branch':<7}{'alpha':>6}{'image':>8}{'fusion':>8}{'gain':>22}"
          f"{'img+rr':>9}{'fus+rr':>9}{'gain with re-ranking':>24}")
    for _, r in R.iterrows():
        sig1 = "*" if (r.desc_lo > 0 or r.desc_hi < 0) else " "
        sig2 = "*" if (r.rr_lo > 0 or r.rr_hi < 0) else " "
        print(f"{r.gallery:<8}{r.branch:<7}{r.alpha_tuned:>6.2f}{r.image_only:>8.2f}{r.fusion:>8.2f}"
              f"{r.desc_gain:>+8.2f} [{r.desc_lo:+.2f},{r.desc_hi:+.2f}]{sig1}"
              f"{r.image_rr:>9.2f}{r.fusion_rr:>9.2f}{r.desc_gain_rr:>+9.2f} [{r.rr_lo:+.2f},{r.rr_hi:+.2f}]{sig2}")
    print("* = interval excludes zero")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = ["native", "mixed", "48px", "32px", "24px", "16px"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
        cmap = plt.cm.viridis(np.linspace(0, .85, len(order)))
        for ax, br, title in zip(axes, ("osnet", "irra"), ("Appearance model (OSNet)", "Vision-language model (IRRA)")):
            for c_, g in zip(cmap, order):
                s = curve[(curve.gallery == g) & (curve.branch == br)]
                ax.plot(s.alpha, s.ap, color=c_, lw=1.6, label=g)
                b = s.loc[s.ap.idxmax()]; ax.plot(b.alpha, b.ap, "o", color=c_, ms=5)
            ax.set_title(title, fontsize=10); ax.set_xlabel("alpha  (0 = description only, 1 = photograph only)")
            ax.set_ylabel("validation mAP, end-to-end (%)"); ax.grid(alpha=.25)
        axes[1].legend(fontsize=8, frameon=False, title="gallery", title_fontsize=8)
        fig.tight_layout(); fig.savefig(out / "alpha_figure.png", dpi=200); plt.close(fig)
        print(f"\nfigure: {out / 'alpha_figure.png'}")
    except Exception as e:
        print("figure skipped:", e)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
