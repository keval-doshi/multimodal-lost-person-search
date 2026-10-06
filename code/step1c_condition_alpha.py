"""
Step 1c - Does quality-aware fusion work at the level of the footage?

Phase 5 found that choosing the balance per CANDIDATE gives nothing. Step 1b
showed that the best balance does shift with resolution when it is chosen
per CONDITION (all footage at a given person size). This script asks the
question that matters: on held-out identities, does choosing alpha for the
resolution condition beat a single fixed alpha?

Three ways of setting alpha, all fitted on VALIDATION identities only:
  fixed (native)   the alpha tuned on normal footage, used everywhere -
                   what a practitioner who tunes once would do
  fixed (pooled)   the single alpha that is best on average across all
                   resolution conditions - the strongest fixed baseline
  condition-aware  a separate alpha for each resolution condition

The condition is observable at search time: it is the typical height of
the detected people in the footage, known from their bounding boxes.

  python step1c_condition_alpha.py --root .\\dataset --limit 60   # smoke
  python step1c_condition_alpha.py --root .\\dataset              # full, ~10 min

Needs cache/step1/alpha_curves.csv from step1b_alpha.py.
Writes cache/step1/condition_results.csv.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from step1_corrected_eval import zrow, unit, rerank, metrics, cboot, LEVELS

CONDITIONS = ["native", "48px", "32px", "24px", "16px"]


def choose(curve, branch):
    """Return (native alpha, pooled alpha, per-condition alphas), ties -> higher alpha."""
    c = curve[(curve.branch == branch) & (curve.gallery.isin(CONDITIONS))]
    def best(s):
        top = s[s.ap >= s.ap.max() - 1e-9]
        return float(top.alpha.max())
    per = {g: best(c[c.gallery == g]) for g in CONDITIONS}
    pooled = c.groupby("alpha").ap.mean().reset_index()
    return per["native"], best(pooled), per


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
    cache = root.parent / "cache"; out = cache / "step1"
    cf = out / "alpha_curves.csv"
    if not cf.exists():
        sys.exit(f"ERROR: {cf} missing - run step1b_alpha.py first.")
    curve = pd.read_csv(cf)

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
    test_ids = set(sh[len(sh) // 2:].tolist())

    test = []
    for qi in qs:
        pid = int(q_pid[qi])
        if pid not in test_ids: continue
        grp = [qi] + [j for j in by_id[pid] if j != qi][:1]
        excl = sorted(set(int(c) for c in q_cam[grp]))
        n_gt = sum(v for (p, c), v in gtc.items() if p == pid and c not in excl)
        if n_gt == 0: continue
        test.append({"qi": qi, "pid": pid, "excl": excl, "n_gt": n_gt,
                     "img": {"osnet": unit(q_os[grp].mean(0)), "irra": unit(q_ir[grp].mean(0))},
                     "txt": unit(c_ir[caps[qi][:2]].mean(0))})
    if a.limit: test = test[:a.limit]
    print(f"held-out queries {len(test)}")

    deg = cache / "degraded"
    Mt = np.load(deg / "meta.npz")
    if not (np.array_equal(Mt["ids"], GP) and np.array_equal(Mt["cams"], GC)):
        sys.exit("ERROR: degraded features are not row-aligned with the gallery.")
    gals = {"native": {"osnet": G["osnet"][mask].astype(np.float32),
                       "irra": G["irra"][mask].astype(np.float32)}}
    for lv in CONDITIONS[1:]:
        f = np.load(deg / f"gallery_{lv}.npz")
        gals[lv] = {"osnet": f["osnet"].astype(np.float32), "irra": f["irra"].astype(np.float32)}

    choices = {br: choose(curve, br) for br in ("osnet", "irra")}
    for br, (na, po, per) in choices.items():
        print(f"  {br}: native alpha {na}, pooled alpha {po}, per condition {per}")

    rows = []
    for g in CONDITIONS:
        Gi = gals[g]["irra"]
        for q in test:
            keep = ~((GP == q["pid"]) & np.isin(GC, q["excl"]))
            idx = np.flatnonzero(keep); rel = GP[idx] == q["pid"]
            zt = zrow(q["txt"] @ Gi[idx].T)
            for br in ("osnet", "irra"):
                feats = gals[g][br][idx]
                zi = zrow(q["img"][br] @ feats.T)
                na, po, per = choices[br]
                for rule, al in (("fixed (native)", na), ("fixed (pooled)", po),
                                 ("condition-aware", per[g])):
                    s = al * zi + (1 - al) * zt
                    for rr in (False, True):
                        sc = rerank(s, feats) if rr else s
                        r = metrics(sc, rel, q["n_gt"])
                        rows.append({"gallery": g, "branch": br, "rule": rule, "alpha": al,
                                     "rerank": rr, "qi": q["qi"], "pid": q["pid"], "ap_e2e": r[1]})
        print(f"  {g} done  [{(time.time() - t0) / 60:.1f} min]")

    df = pd.DataFrame(rows)
    df.to_csv(out / "condition_perquery.csv", index=False)

    res = []
    for (g, br, rr), s in df.groupby(["gallery", "branch", "rerank"], sort=False):
        def get(rule): return s[s.rule == rule].set_index("qi")
        ca = get("condition-aware")
        for base in ("fixed (native)", "fixed (pooled)"):
            b = get(base)
            j = ca.join(b[["ap_e2e", "alpha"]], rsuffix="_b")
            m, lo, hi, _ = cboot((j.ap_e2e - j.ap_e2e_b).values * 100, j.pid.values)
            res.append({"gallery": g, "branch": br, "rerank": rr, "baseline": base,
                        "alpha_condition": ca.alpha.iloc[0], "alpha_baseline": b.alpha.iloc[0],
                        "map_condition": round(ca.ap_e2e.mean() * 100, 2),
                        "map_baseline": round(b.ap_e2e.mean() * 100, 2),
                        "diff": round(m, 2), "lo": round(lo, 2), "hi": round(hi, 2),
                        "significant": bool(lo > 0 or hi < 0)})
    R = pd.DataFrame(res)
    R.to_csv(out / "condition_results.csv", index=False)

    # pooled across conditions: one overall test per branch and baseline
    print("\n" + "=" * 96)
    print("CONDITION-AWARE vs FIXED alpha  (held-out identities, end-to-end mAP, with re-ranking)")
    print("=" * 96)
    for br in ("osnet", "irra"):
        print(f"\n  {br}")
        x = R[(R.branch == br) & (R.rerank)]
        for _, r in x.iterrows():
            same = " (same alpha)" if r.alpha_condition == r.alpha_baseline else ""
            print(f"    {r.gallery:<7} vs {r.baseline:<15} a={r.alpha_condition:.2f} vs {r.alpha_baseline:.2f}"
                  f"  {r.map_condition:6.2f} vs {r.map_baseline:6.2f}  {r['diff']:+6.2f} "
                  f"[{r.lo:+.2f},{r.hi:+.2f}] {'SIG' if r.significant else 'ns'}{same}")
        for base in ("fixed (native)", "fixed (pooled)"):
            s = df[(df.branch == br) & (df.rerank)]
            ca = s[s.rule == "condition-aware"].set_index(["gallery", "qi"])
            b = s[s.rule == base].set_index(["gallery", "qi"])
            j = ca.join(b[["ap_e2e"]], rsuffix="_b")
            m, lo, hi, _ = cboot((j.ap_e2e - j.ap_e2e_b).values * 100, j.pid.values)
            print(f"    ALL CONDITIONS vs {base:<15} {m:+6.2f} [{lo:+.2f},{hi:+.2f}] "
                  f"{'SIG' if (lo > 0 or hi < 0) else 'ns'}")
    print(f"\nsaved {out / 'condition_results.csv'}   [{(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
