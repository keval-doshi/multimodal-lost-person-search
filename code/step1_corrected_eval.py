"""
Step 1 - Corrected evaluation.

Replaces the headline evaluation of Phases 7 and 8. It fixes the problems
found in review and reports every number the results chapter needs.

  # smoke test, about two minutes
  python step1_corrected_eval.py --root .\\dataset --limit 60

  # full run
  python step1_corrected_eval.py --root .\\dataset

What changed, and why
---------------------
1  RE-RANKING TAIL. The old code gave every candidate outside the re-ranked
   top 200 the same score (-1e9), so their order was arbitrary and correct
   matches below rank 200 contributed a random amount to mAP. Now the top
   200 are re-ordered and everything below keeps its original order.

2  THE RECOMMENDED SYSTEM IS RUN EXPLICITLY. The old headline figures were
   read off a "remove one component" row of a larger system. Now the four-
   component system (fusion, two descriptions, two photographs, re-ranking)
   is evaluated directly, and each component is removed from THAT system.
   Query expansion and resolution matching are reported as add-backs.

3  END-TO-END mAP. The old AP counted only people the detector found, and
   searches where detection failed completely were skipped. Two measures
   are now reported:
     conditional   AP over detected people only (the old measure)
     end-to-end    AP x (detected / annotated), so people the detector
                   missed count against the result, and a search whose
                   target was never detected scores zero. This follows the
                   CUHK-SYSU / PRW person-search convention.

4  STANDARD JUNK RULE. The old code removed every candidate from the query's
   camera. The standard rule removes only candidates that are BOTH the
   query identity AND from the query camera(s); other people seen by that
   camera stay as distractors. --protocol crosscam reproduces the old rule.

5  FAIR BASELINES. Appearance-only baselines are reported for both encoders
   (OSNet and IRRA) with one and with two photographs, plus description-only
   with one and two descriptions, all under the same queries and exclusions.
   The recommended system is also built on the IRRA image branch, and the
   gain is split into "extra inputs" and "algorithm".

6  IDENTITY-LEVEL BOOTSTRAP. Several queries belong to the same person, so
   they are not independent. Intervals now resample IDENTITIES (all of a
   person's queries together), with 5,000 resamples everywhere. Differences
   between methods are paired: the same resampled identities for both.

Parameters are the ones fitted on the validation identities in Phases 5
and 7; they are not re-tuned here. Only held-out identities are evaluated.
Also written: detection recall AFTER the 0.5 gallery filter, by height.

Outputs (in cache/step1/)
  perquery.csv        every query x method x gallery, for re-analysis
  summary.csv         mAP and Rank-1 with intervals
  comparisons.csv     paired differences with intervals
  recall_filtered.csv detection recall of the actual gallery
  config.json         every parameter used
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

LEVELS = ["native", "96px", "64px", "48px", "32px", "24px", "16px"]
BOUNDS = [20, 28, 40, 56, 80, 128]
ORDER = ["16px", "24px", "32px", "48px", "64px", "96px", "native"]

ALPHA_OSNET = 0.45      # tuned on validation identities, Phase 5
ALPHA_IRRA = 0.80       # tuned on validation identities, Phase 5
KRECIP = (20, 6, 0.3)   # tuned on validation identities, Phase 7
TOPN = 200
QE_K = 8
REPS = 5000


# ============================================================ primitives
def zrow(v):
    return (v - v.mean()) / (v.std() + 1e-8)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def k_reciprocal(sub_sim, q_sim, k1=20, k2=6, lam=0.3):
    """
    K-reciprocal re-ranking (Zhong et al., 2017) on a candidate shortlist.
    Same algorithm and blend as the Phase 8 version, vectorised; the unit
    tests check it against the original line by line.
    """
    N = len(q_sim)
    if N < 4:
        return np.asarray(q_sim, np.float64).copy()
    k1 = min(k1, N - 1)
    k2 = max(1, min(k2, k1))
    kh = int(round(k1 / 2))
    full = np.empty((N + 1, N + 1), np.float32)
    full[0, 1:] = q_sim
    full[1:, 0] = q_sim
    full[1:, 1:] = sub_sim
    np.fill_diagonal(full, 1.0)
    order = np.argsort(-full, axis=1)
    rows = np.arange(N + 1)[:, None]

    def recip(k):
        M = np.zeros((N + 1, N + 1), bool)
        M[rows, order[:, :k + 1]] = True
        R = M & M.T
        np.fill_diagonal(R, True)
        return R

    Rk, Rh = recip(k1), recip(kh)
    hsize = Rh.sum(1)
    V = np.zeros((N + 1, N + 1), np.float32)
    for i in range(N + 1):
        mem = np.flatnonzero(Rk[i])
        overlap = (Rh[mem] & Rk[i]).sum(1)
        grow = mem[overlap >= (2.0 / 3.0) * hsize[mem]]
        Rs = Rk[i].copy()
        if len(grow):
            Rs |= Rh[grow].any(0)
        idx = np.flatnonzero(Rs)
        w = np.exp(full[i, idx] - full[i, idx].max())
        V[i, idx] = w / (w.sum() + 1e-12)
    if k2 > 1:
        V = V[order[:, :k2]].mean(axis=1)
    vq = V[0]
    inter = np.minimum(vq[None, :], V[1:]).sum(1)
    union = np.maximum(vq[None, :], V[1:]).sum(1)
    return (1 - lam) * (inter / (union + 1e-12)) + lam * np.asarray(q_sim)


def rerank(scores, feats, topn=TOPN, params=KRECIP):
    """
    FIX 1. Re-order the top `topn` by k-reciprocal distance and keep every
    candidate below them in its ORIGINAL order. Returns rank-based scores,
    so there are no ties anywhere in the list.
    """
    n = min(topn, len(scores))
    order = np.argsort(-scores, kind="stable")
    top, rest = order[:n], order[n:]
    sub = feats[top]
    r = k_reciprocal(sub @ sub.T, scores[top], *params)
    final = np.concatenate([top[np.argsort(-r, kind="stable")], rest])
    out = np.empty(len(scores), np.float64)
    out[final] = -np.arange(len(scores), dtype=np.float64)
    return out


def query_expansion(q, g, s, k=QE_K, power=3.0):
    top = np.argsort(-s)[:k]
    w = np.clip(s[top], 0, None) ** power
    if w.sum() < 1e-9:
        return q
    return unit(q + (g[top] * w[:, None]).sum(0) / w.sum())


def metrics(scores, rel, n_gt):
    """
    FIX 3. Returns (ap_conditional, ap_end_to_end, r1_conditional, r1_end_to_end).
    Conditional ignores missed people; end-to-end scales by detection recall
    and scores a search whose target was never detected as zero.
    """
    if n_gt <= 0:
        return None
    n_rel = int(rel.sum())
    if n_rel == 0:
        return (np.nan, 0.0, np.nan, 0.0)
    o = np.argsort(-scores, kind="stable")
    ro = rel[o]
    pos = np.flatnonzero(ro)
    ap = float(np.mean((np.arange(n_rel) + 1) / (pos + 1)))
    r1 = float(ro[0])
    recall = min(1.0, n_rel / n_gt)
    return (ap, ap * recall, r1, r1)


def cboot(vals, pids, reps=REPS, seed=0):
    """
    FIX 6. Identity-level (cluster) bootstrap of a mean. All queries of a
    resampled identity enter together. NaNs are dropped first.
    """
    vals = np.asarray(vals, float)
    pids = np.asarray(pids)
    ok = ~np.isnan(vals)
    v, p = vals[ok], pids[ok]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan, 0)
    u, inv = np.unique(p, return_inverse=True)
    sums = np.bincount(inv, weights=v)
    cnt = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(reps, len(u)))
    means = sums[idx].sum(1) / cnt[idx].sum(1)
    return (v.mean(), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), len(u))


# ============================================================ main
def main():
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--root", default="dataset")
    ap_.add_argument("--protocol", default="standard", choices=["standard", "crosscam"])
    ap_.add_argument("--seed", type=int, default=0)
    ap_.add_argument("--limit", type=int, default=0)
    ap_.add_argument("--no-resmatch", action="store_true",
                     help="skip the resolution-matching add-back explicitly")
    a = ap_.parse_args()

    from phase3_baseline import load_split
    from phase3_encode import load_gt_boxes, iou_matrix

    root = Path(a.root).resolve()
    prw = root / "PRW"
    cache = root.parent / "cache"
    out = cache / "step1"
    out.mkdir(exist_ok=True)
    t0 = time.time()

    G = np.load(cache / "gallery.npz", allow_pickle=True)
    Q = np.load(cache / "query.npz", allow_pickle=True)
    C = np.load(cache / "captions.npz", allow_pickle=True)
    test_frames = load_split(prw, "frame_test.mat")
    mask = np.array([f in test_frames for f in G["frames"]])
    g_pid, g_cam, g_h = G["ids"][mask], G["cams"][mask], G["height"][mask]
    g_frames, g_boxes, g_conf = G["frames"][mask], G["boxes"][mask], G["conf"][mask]
    g_os = G["osnet"][mask].astype(np.float32)
    g_ir = G["irra"][mask].astype(np.float32)

    # ---- resolution-matching assets: required unless explicitly skipped
    qd_path = cache / "query_degraded.npz"
    use_rm = not a.no_resmatch
    if use_rm and not qd_path.exists():
        sys.exit(f"ERROR: {qd_path} missing. Run phase7_resmatch.py --encode first, "
                 f"or pass --no-resmatch to skip that add-back explicitly.")
    QD = np.load(qd_path, allow_pickle=True) if use_rm else None

    # ---- ground truth: annotated appearances per (identity, camera)
    print("Reading test annotations ...")
    frames_sorted = sorted(test_frames)
    gt = load_gt_boxes(prw, frames_sorted)
    gt_count = {}
    for f, (boxes, ids) in gt.items():
        cam = int(f.split("s")[0][1:])
        for pid in ids:
            if pid > 0:
                gt_count[(int(pid), cam)] = gt_count.get((int(pid), cam), 0) + 1

    # ---- detection recall of the ACTUAL gallery (after the 0.5 filter)
    print("Detection recall after the gallery filter ...")
    by_frame = {}
    for i, f in enumerate(g_frames):
        by_frame.setdefault(str(f), []).append(i)
    bands = [(0, 64), (64, 96), (96, 128), (128, 192), (192, 256), (256, 10 ** 6)]
    tot = np.zeros(len(bands)); hit = np.zeros(len(bands))
    for f, (gb, _) in gt.items():
        if len(gb) == 0:
            continue
        heights = gb[:, 3] - gb[:, 1]
        det = by_frame.get(f, [])
        matched = np.zeros(len(gb), bool)
        if det:
            db = g_boxes[det].astype(float)
            M = iou_matrix(db, gb)
            for di in np.argsort(-g_conf[det]):
                gj = int(np.argmax(M[di]))
                if M[di, gj] >= 0.5 and not matched[gj]:
                    matched[gj] = True
        for k, (lo, hi) in enumerate(bands):
            sel = (heights >= lo) & (heights < hi)
            tot[k] += sel.sum(); hit[k] += (sel & matched).sum()
    rec_rows = [{"height": f"{lo}-{hi if hi < 10**6 else '+'}", "annotated": int(t),
                 "recall_pct": round(h / t * 100, 1) if t else None}
                for (lo, hi), t, h in zip(bands, tot, hit)]
    rec_rows.append({"height": "overall", "annotated": int(tot.sum()),
                     "recall_pct": round(hit.sum() / tot.sum() * 100, 1)})
    pd.DataFrame(rec_rows).to_csv(out / "recall_filtered.csv", index=False)
    for r in rec_rows:
        print(f"  {r['height']:<10}{r['annotated']:>7}  {r['recall_pct']}%")

    # ---- queries and the identity-disjoint split (same draw as Phases 5-8)
    q_os, q_ir = Q["osnet"].astype(np.float32), Q["irra"].astype(np.float32)
    q_pid, q_cam, q_frames = Q["ids"], Q["cams"], Q["frames"]
    c_ir = C["irra"].astype(np.float32)
    key = {(int(p), str(f)): i for i, (p, f) in enumerate(zip(q_pid, q_frames))}
    caps_of = {}
    for i, (p, f) in enumerate(zip(C["ids"], C["frames"])):
        r = key.get((int(p), str(f)))
        if r is not None:
            caps_of.setdefault(r, []).append(i)
    by_id = {}
    for i in range(len(q_pid)):
        if i in caps_of:
            by_id.setdefault(int(q_pid[i]), []).append(i)
    qs = [i for i in range(len(q_pid)) if i in caps_of]
    ids = np.unique(q_pid[qs])
    rng = np.random.default_rng(a.seed)
    sh = ids.copy(); rng.shuffle(sh)
    test_ids = set(sh[len(sh) // 2:].tolist())
    tq = [i for i in qs if int(q_pid[i]) in test_ids]
    if a.limit:
        tq = tq[:a.limit]
    print(f"\nHeld-out queries: {len(tq)} from {len(test_ids)} identities "
          f"(protocol: {a.protocol})")

    q_os_lv = ({lv: QD[f"os_{lv}"].astype(np.float32) for lv in LEVELS}
               if use_rm else None)

    # ---- gallery configurations
    galleries = {"native": (g_os, g_ir, g_pid, g_cam,
                            np.array([ORDER[int(np.digitize(h, BOUNDS))] for h in g_h]))}
    deg = cache / "degraded"
    if (deg / "meta.npz").exists():
        Mt = np.load(deg / "meta.npz")
        if not (np.array_equal(Mt["ids"], g_pid) and np.array_equal(Mt["cams"], g_cam)):
            sys.exit("ERROR: degraded features are not row-aligned with the gallery.")
        lf = {lv: np.load(deg / f"gallery_{lv}.npz") for lv in LEVELS}
        pick = np.random.default_rng(a.seed).integers(0, len(LEVELS), size=len(g_pid))
        gm = np.zeros_like(g_os); gmi = np.zeros_like(g_ir)
        lvm = np.empty(len(g_pid), object)
        for k, lv in enumerate(LEVELS):
            m = pick == k
            gm[m] = lf[lv]["osnet"][m].astype(np.float32)
            gmi[m] = lf[lv]["irra"][m].astype(np.float32)
            lvm[m] = lv
        galleries["mixed"] = (gm, gmi, g_pid, g_cam, lvm.astype(str))
    else:
        print("NOTE: degraded features missing; only the native gallery is evaluated.")

    records = []
    for gname, (G_os, G_ir, GP, GC, GL) in galleries.items():
        print(f"\n=== gallery: {gname} ===")
        for n_done, qi in enumerate(tq, 1):
            pid = int(q_pid[qi])
            group = [qi] + [j for j in by_id[pid] if j != qi][:1]
            excl = sorted(set(int(c) for c in q_cam[group]))
            # FIX 4: junk = same identity in an excluded camera
            if a.protocol == "standard":
                keep = ~((GP == pid) & np.isin(GC, excl))
            else:
                keep = ~np.isin(GC, excl)
            idx = np.flatnonzero(keep)
            rel = GP[idx] == pid
            n_gt = sum(v for (p, c), v in gt_count.items() if p == pid and c not in excl)
            if n_gt == 0:
                continue
            go, gi, lvl = G_os[idx], G_ir[idx], GL[idx]

            cl = caps_of[qi]
            t1 = zrow(c_ir[cl[0]] @ gi.T)
            t2 = zrow(unit(c_ir[cl[:2]].mean(0)) @ gi.T) if len(cl) > 1 else t1
            multi = len(group) > 1
            qo1, qo2 = q_os[qi], (unit(q_os[group].mean(0)) if multi else q_os[qi])
            qi1, qi2 = q_ir[qi], (unit(q_ir[group].mean(0)) if multi else q_ir[qi])
            o1, o2 = zrow(qo1 @ go.T), zrow(qo2 @ go.T)
            i1, i2 = zrow(qi1 @ gi.T), zrow(qi2 @ gi.T)

            A, B = ALPHA_OSNET, ALPHA_IRRA
            fuse1 = A * o1 + (1 - A) * t1              # single inputs
            rec = A * o2 + (1 - A) * t2                # recommended, before re-ranking

            S = {
                "OSNet photo x1": o1, "OSNet photos x2": o2,
                "IRRA photo x1": i1, "IRRA photos x2": i2,
                "Description x1": t1, "Descriptions x2": t2,
                "Fusion, single inputs": fuse1,
                "Fusion, single inputs + re-ranking": rerank(fuse1, go),
                "Recommended": rerank(rec, go),
                "Recommended - fusion": rerank(o2, go),
                "Recommended - second description": rerank(A * o2 + (1 - A) * t1, go),
                "Recommended - second photograph": rerank(A * o1 + (1 - A) * t2, go),
                "Recommended - re-ranking": rec,
                "Recommended (IRRA image branch)": rerank(B * i2 + (1 - B) * t2, gi),
                "IRRA fusion, no re-ranking": B * i2 + (1 - B) * t2,
            }
            qv = query_expansion(qo2, go, rec)
            S["Recommended + query expansion"] = rerank(A * zrow(qv @ go.T) + (1 - A) * t2, go)
            if use_rm:
                zr = np.empty(len(idx), np.float32)
                for lv in LEVELS:
                    m = lvl == lv
                    if m.any():
                        ql = unit(q_os_lv[lv][group].mean(0)) if multi else q_os_lv[lv][qi]
                        zr[m] = zrow(ql @ go.T)[m]
                S["Recommended + resolution matching"] = rerank(A * zr + (1 - A) * t2, go)

            for meth, sc in S.items():
                r = metrics(sc, rel, n_gt)
                records.append({"gallery": gname, "method": meth, "qi": qi, "pid": pid,
                                "n_gt": n_gt, "n_detected": int(rel.sum()),
                                "ap_cond": r[0], "ap_e2e": r[1],
                                "r1_cond": r[2], "r1_e2e": r[3]})
            if n_done % 100 == 0:
                el = time.time() - t0
                print(f"  {n_done}/{len(tq)} queries  [{el / 60:.1f} min]")

    df = pd.DataFrame(records)
    df.to_csv(out / "perquery.csv", index=False)

    # ---------------------------------------------------- summary
    rows = []
    for (gname, meth), s in df.groupby(["gallery", "method"], sort=False):
        row = {"gallery": gname, "method": meth, "queries": len(s),
               "identities": s.pid.nunique(),
               "no_target_detected": int((s.n_detected == 0).sum())}
        for col in ("ap_cond", "ap_e2e", "r1_e2e"):
            m, lo, hi, _ = cboot(s[col].values * 100, s.pid.values)
            row[col] = round(m, 2); row[col + "_lo"] = round(lo, 2); row[col + "_hi"] = round(hi, 2)
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary.csv", index=False)

    # ---------------------------------------------------- paired comparisons
    def paired(g, m1, m2, col):
        a1 = df[(df.gallery == g) & (df.method == m1)].set_index("qi")
        a2 = df[(df.gallery == g) & (df.method == m2)].set_index("qi")
        j = a1.join(a2[[col]], rsuffix="_b", how="inner")
        d = (j[col] - j[col + "_b"]).values * 100
        m, lo, hi, n = cboot(d, j.pid.values)
        return m, lo, hi, n

    comps = [
        ("Recommended", "OSNet photo x1", "system vs OSNet photo (single)"),
        ("Recommended", "IRRA photo x1", "system vs IRRA photo (single)"),
        ("Recommended", "OSNet photos x2", "system vs OSNet photos (two)"),
        ("Recommended", "IRRA photos x2", "system vs IRRA photos (two)"),
        ("Recommended (IRRA image branch)", "IRRA photos x2", "IRRA system vs IRRA photos (two)"),
        ("Fusion, single inputs", "OSNet photo x1", "fusion alone, same inputs"),
        ("Fusion, single inputs + re-ranking", "Fusion, single inputs", "re-ranking, same inputs"),
        ("Recommended - re-ranking", "Fusion, single inputs", "extra inputs only"),
        ("Recommended - fusion", "Recommended", "remove fusion"),
        ("Recommended - second description", "Recommended", "remove second description"),
        ("Recommended - second photograph", "Recommended", "remove second photograph"),
        ("Recommended - re-ranking", "Recommended", "remove re-ranking"),
        ("Recommended + query expansion", "Recommended", "add query expansion"),
        ("Recommended + resolution matching", "Recommended", "add resolution matching"),
    ]
    crow = []
    for g in df.gallery.unique():
        for m1, m2, lab in comps:
            if m1 not in set(df.method) or m2 not in set(df.method):
                continue
            for col in ("ap_e2e", "ap_cond"):
                m, lo, hi, n = paired(g, m1, m2, col)
                crow.append({"gallery": g, "comparison": lab, "metric": col,
                             "diff": round(m, 2), "lo": round(lo, 2), "hi": round(hi, 2),
                             "significant": bool(lo > 0 or hi < 0), "identities": n})
    cmp = pd.DataFrame(crow)
    cmp.to_csv(out / "comparisons.csv", index=False)

    json.dump({"protocol": a.protocol, "alpha_osnet": ALPHA_OSNET, "alpha_irra": ALPHA_IRRA,
               "krecip_k1_k2_lambda": KRECIP, "rerank_topn": TOPN, "qe_k": QE_K,
               "bootstrap": f"{REPS} resamples, identity-level", "seed": a.seed,
               "held_out_queries": len(tq), "held_out_identities": len(test_ids),
               "resolution_matching": use_rm, "limit": a.limit},
              open(out / "config.json", "w"), indent=2)

    # ---------------------------------------------------- print
    pd.set_option("display.width", 200)
    for g in summ.gallery.unique():
        s = summ[summ.gallery == g]
        print(f"\n{'=' * 92}\nRESULTS - {g} gallery  (identity-level 95% intervals)\n{'=' * 92}")
        print(f"{'method':<38}{'end-to-end mAP':>22}{'conditional mAP':>22}{'R1 e2e':>9}")
        for _, r in s.iterrows():
            print(f"{r.method:<38}{r.ap_e2e:>7.2f} [{r.ap_e2e_lo:5.2f},{r.ap_e2e_hi:5.2f}]"
                  f"{r.ap_cond:>9.2f} [{r.ap_cond_lo:5.2f},{r.ap_cond_hi:5.2f}]{r.r1_e2e:>9.2f}")
        c = cmp[(cmp.gallery == g) & (cmp.metric == "ap_e2e")]
        print(f"\npaired differences, end-to-end mAP")
        for _, r in c.iterrows():
            print(f"  {r.comparison:<36}{r['diff']:+7.2f} [{r.lo:+.2f}, {r.hi:+.2f}]"
                  f"  {'significant' if r.significant else 'not significant'}")
    print(f"\nSaved to {out}   [{(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
