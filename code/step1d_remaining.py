"""
Step 1d - Re-run the remaining Chapter 5 analyses under the corrected protocol.

Steps 1, 1b and 1c covered the complete system, the fusion balance and the
condition-level balance. Four analyses in Chapter 5 still rested on the old
protocol (every same-camera candidate removed, conditional mAP only, query-
level intervals, and in some cases all queries including tuning identities).
This script redoes them so that every number in Chapter 5 comes from one
protocol:

  A  RESOLUTION STUDY (RQ3). One photograph against one description at seven
     resolution levels, both encoders, held-out identities, paired identity-
     level intervals. Replaces the Phase 4 crossing-point table.
  B  PER-CANDIDATE QUALITY WEIGHTING. The Phase 5 rule (balance varies with
     each candidate's resolution) refitted on validation identities and
     tested against a tuned constant on the mixed gallery.
  C  EXPLORATORY ANALYSES on held-out identities: overlap of what the photo
     and the description find, sensitivity to which of the two independently
     written descriptions is used, and camera coverage against ground truth.
  D  CONFIDENCE. Quintile thresholds fixed on validation identities, then
     applied to held-out identities.

  python step1d_remaining.py --root .\\dataset --limit 60   # smoke, ~3 min
  python step1d_remaining.py --root .\\dataset              # full, ~25 min

Writes everything to cache/step1/d_*.csv and d_summary.txt.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from step1_corrected_eval import zrow, unit, rerank, metrics, cboot, LEVELS

LEVEL_PX = {"native": 10 ** 6, "96px": 96, "64px": 64, "48px": 48,
            "32px": 32, "24px": 24, "16px": 16}


# ---------------------------------------------------------------- helpers
def keep_idx(GP, GC, pid, excl):
    return np.flatnonzero(~((GP == pid) & np.isin(GC, excl)))


def paired(a, b, pids):
    """paired identity-level difference of two per-query arrays (x100)."""
    d = (np.asarray(a, float) - np.asarray(b, float)) * 100
    return cboot(d, np.asarray(pids))


def fmt(m, lo, hi):
    return f"{m:+6.2f} [{lo:+.2f},{hi:+.2f}]{' *' if (lo > 0 or hi < 0) else '  '}"


# ---------------------------------------------------------------- A
def resolution_study(gals, GP, GC, test):
    rows = []
    for lv in LEVELS:
        Gi = gals[lv]
        for q in test:
            idx = keep_idx(GP, GC, q["pid"], q["excl1"])
            rel = GP[idx] == q["pid"]
            zt = zrow(q["txt1"] @ Gi["irra"][idx].T)
            rec = {"level": lv, "qi": q["qi"], "pid": q["pid"]}
            for name, s in (("photo_osnet", zrow(q["os1"] @ Gi["osnet"][idx].T)),
                            ("photo_irra", zrow(q["ir1"] @ Gi["irra"][idx].T)),
                            ("text", zt)):
                r = metrics(s, rel, q["n_gt1"])
                rec[f"{name}_e2e"], rec[f"{name}_cond"], rec[f"{name}_r1"] = r[1], r[0], r[3]
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- B
def per_candidate(gal_mixed, lvl, GP, GC, val, test, branch, alpha_const,
                  centres=np.round(np.arange(0, 1.001, 0.05), 2),
                  slopes=np.round(np.arange(-1.0, 1.001, 0.25), 2)):
    """alpha_i = clip(centre + slope * (q_i - 0.5)); slope 0 == constant."""
    px = np.array([min(LEVEL_PX[l], 160) for l in lvl], np.float32)
    qual = np.clip((px - 16) / (160 - 16), 0, 1)
    def prep(qs):
        out = []
        for q in qs:
            idx = keep_idx(GP, GC, q["pid"], q["excl"])
            rel = GP[idx] == q["pid"]
            zi = zrow(q["img"][branch] @ gal_mixed[branch][idx].T)
            zt = zrow(q["txt"] @ gal_mixed["irra"][idx].T)
            out.append((idx, rel, zi, zt, q["n_gt"], q["pid"]))
        return out
    V, Tt = prep(val), prep(test)
    def score(data, c, s):
        aps = []
        for idx, rel, zi, zt, ng, _ in data:
            a = np.clip(c + s * (qual[idx] - 0.5), 0, 1)
            aps.append(metrics(a * zi + (1 - a) * zt, rel, ng)[1])
        return np.array(aps)
    best, bc, bs = -1, alpha_const, 0.0
    for c in centres:
        for s in slopes:
            m = np.nanmean(score(V, c, s))
            if m > best + 1e-12 or (abs(m - best) <= 1e-12 and s == 0):
                best, bc, bs = m, float(c), float(s)
    rule = score(Tt, bc, bs)
    const = score(Tt, alpha_const, 0.0)
    pids = [d[5] for d in Tt]
    return {"branch": branch, "centre": bc, "slope": bs, "alpha_const": alpha_const,
            "rule": np.nanmean(rule) * 100, "const": np.nanmean(const) * 100,
            "diff": paired(rule, const, pids)}


# ---------------------------------------------------------------- C + D
def exploratory(G_native, GP, GC, GCam_gt, test, val, alpha_os, alpha_ir):
    rows = []
    def run(qs):
        out = []
        for q in qs:
            idx = keep_idx(GP, GC, q["pid"], q["excl"])
            rel = GP[idx] == q["pid"]
            gcam = GC[idx]
            zi_os = zrow(q["img"]["osnet"] @ G_native["osnet"][idx].T)
            zt1 = zrow(q["txt1"] @ G_native["irra"][idx].T)
            zt2 = zrow(q["txt2"] @ G_native["irra"][idx].T) if q["txt2"] is not None else None
            zt = zrow(q["txt"] @ G_native["irra"][idx].T)
            fused = alpha_os * zi_os + (1 - alpha_os) * zt
            fr = rerank(fused, G_native["osnet"][idx])
            top_i = set(np.argsort(-zi_os)[:10][rel[np.argsort(-zi_os)[:10]]])
            top_t = set(np.argsort(-zt)[:10][rel[np.argsort(-zt)[:10]]])
            o20 = np.argsort(-fr)[:20]
            found = set(gcam[o20][rel[o20]].tolist())
            true_cams = GCam_gt.get(q["pid"], set()) - set(q["excl"])
            sf = np.sort(fr)[::-1]
            fs = np.sort(fused)[::-1]
            rec = {"qi": q["qi"], "pid": q["pid"],
                   "hit_img": len(top_i), "hit_txt": len(top_t),
                   "overlap": len(top_i & top_t), "union": len(top_i | top_t),
                   "ap_txt1": metrics(zt1, rel, q["n_gt"])[1],
                   "ap_txt2": metrics(zt2, rel, q["n_gt"])[1] if zt2 is not None else np.nan,
                   "ap_fus1": metrics(alpha_os * zi_os + (1 - alpha_os) * zt1, rel, q["n_gt"])[1],
                   "ap_fus2": (metrics(alpha_os * zi_os + (1 - alpha_os) * zt2, rel, q["n_gt"])[1]
                               if zt2 is not None else np.nan),
                   "ap_txt_avg": metrics(zt, rel, q["n_gt"])[1],
                   "cap_len": q["cap_len"],
                   "cams_true": len(true_cams), "cams_found": len(found & true_cams),
                   "conf": float(fs[0] - fs[1]),
                   "r1": metrics(fr, rel, q["n_gt"])[3]}
            out.append(rec)
        return pd.DataFrame(out)
    return run(test), run(val)


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
    curves = pd.read_csv(out / "alpha_curves.csv")

    G = np.load(cache / "gallery.npz", allow_pickle=True)
    Q = np.load(cache / "query.npz", allow_pickle=True)
    C = np.load(cache / "captions.npz", allow_pickle=True)
    tf = load_split(prw, "frame_test.mat")
    mask = np.array([f in tf for f in G["frames"]])
    GP, GC = G["ids"][mask], G["cams"][mask]

    gt = load_gt_boxes(prw, sorted(tf))
    gtc, gcams = {}, {}
    for f, (_, ids) in gt.items():
        cam = int(f.split("s")[0][1:])
        for p in ids:
            if p > 0:
                gtc[(int(p), cam)] = gtc.get((int(p), cam), 0) + 1
                gcams.setdefault(int(p), set()).add(cam)

    q_os, q_ir = Q["osnet"].astype(np.float32), Q["irra"].astype(np.float32)
    q_pid, q_cam, q_frames = Q["ids"], Q["cams"], Q["frames"]
    c_ir = C["irra"].astype(np.float32); c_txt = [str(t) for t in C["text"]]
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
    val_ids = set(sh[:len(sh) // 2].tolist())

    def ngt(pid, excl): return sum(v for (p, c), v in gtc.items() if p == pid and c not in excl)
    val, test = [], []
    for qi in qs:
        pid = int(q_pid[qi])
        grp = [qi] + [j for j in by_id[pid] if j != qi][:1]
        excl = sorted(set(int(c) for c in q_cam[grp])); excl1 = [int(q_cam[qi])]
        n2, n1 = ngt(pid, excl), ngt(pid, excl1)
        if n2 == 0 or n1 == 0: continue
        cl = caps[qi]
        q = {"qi": qi, "pid": pid, "excl": excl, "excl1": excl1, "n_gt": n2, "n_gt1": n1,
             "img": {"osnet": unit(q_os[grp].mean(0)), "irra": unit(q_ir[grp].mean(0))},
             "os1": q_os[qi], "ir1": q_ir[qi],
             "txt": unit(c_ir[cl[:2]].mean(0)), "txt1": c_ir[cl[0]],
             "txt2": c_ir[cl[1]] if len(cl) > 1 else None,
             "cap_len": len(c_txt[cl[0]].split())}
        (val if pid in val_ids else test).append(q)
    if a.limit: val, test = val[:a.limit], test[:a.limit]
    print(f"validation {len(val)} | held-out {len(test)} queries")

    deg = cache / "degraded"
    Mt = np.load(deg / "meta.npz")
    if not (np.array_equal(Mt["ids"], GP) and np.array_equal(Mt["cams"], GC)):
        sys.exit("ERROR: degraded features are not row-aligned with the gallery.")
    lf = {lv: np.load(deg / f"gallery_{lv}.npz") for lv in LEVELS}
    gals = {lv: {"osnet": lf[lv]["osnet"].astype(np.float32),
                 "irra": lf[lv]["irra"].astype(np.float32)} for lv in LEVELS}
    gals["native"] = {"osnet": G["osnet"][mask].astype(np.float32),
                      "irra": G["irra"][mask].astype(np.float32)}
    pick = np.random.default_rng(a.seed).integers(0, len(LEVELS), size=len(GP))
    mixed = {"osnet": np.zeros_like(gals["native"]["osnet"]),
             "irra": np.zeros_like(gals["native"]["irra"])}
    lvl = np.empty(len(GP), object)
    for k, lv in enumerate(LEVELS):
        m = pick == k
        for br in mixed: mixed[br][m] = gals[lv][br][m]
        lvl[m] = lv
    lvl = lvl.astype(str)

    def alpha_of(g, br):
        s = curves[(curves.gallery == g) & (curves.branch == br)]
        top = s[s.ap >= s.ap.max() - 1e-9]; return float(top.alpha.max())
    a_os_n, a_ir_n = alpha_of("native", "osnet"), alpha_of("native", "irra")
    a_os_m, a_ir_m = alpha_of("mixed", "osnet"), alpha_of("mixed", "irra")

    lines = []
    def say(s=""):
        print(s); lines.append(s)

    # ---------------- A
    A = resolution_study(gals, GP, GC, test)
    A.to_csv(out / "d_resolution_perquery.csv", index=False)
    say("=" * 100)
    say("A  RESOLUTION STUDY - one photograph vs one description, held-out identities, end-to-end mAP")
    say("   difference = photograph minus description, paired identity-level 95% interval (* excludes 0)")
    say("=" * 100)
    say(f"{'level':<8}{'OSNet':>8}{'IRRA':>8}{'desc':>8}{'OSNet - desc':>26}{'IRRA - desc':>26}")
    resA = []
    for lv in LEVELS:
        s = A[A.level == lv]
        d1 = paired(s.photo_osnet_e2e, s.text_e2e, s.pid)
        d2 = paired(s.photo_irra_e2e, s.text_e2e, s.pid)
        mo = cboot(s.photo_osnet_e2e.values * 100, s.pid.values)
        mi = cboot(s.photo_irra_e2e.values * 100, s.pid.values)
        mt = cboot(s.text_e2e.values * 100, s.pid.values)
        say(f"{lv:<8}{mo[0]:>8.2f}{mi[0]:>8.2f}{mt[0]:>8.2f}  {fmt(*d1[:3])}  {fmt(*d2[:3])}")
        resA.append({"level": lv, "osnet": mo[0], "osnet_lo": mo[1], "osnet_hi": mo[2],
                     "irra": mi[0], "irra_lo": mi[1], "irra_hi": mi[2],
                     "text": mt[0], "text_lo": mt[1], "text_hi": mt[2],
                     "os_minus_text": d1[0], "os_lo": d1[1], "os_hi": d1[2],
                     "ir_minus_text": d2[0], "ir_lo": d2[1], "ir_hi": d2[2],
                     "osnet_r1": s.photo_osnet_r1.mean() * 100, "irra_r1": s.photo_irra_r1.mean() * 100,
                     "text_r1": s.text_r1.mean() * 100,
                     "osnet_cond": np.nanmean(s.photo_osnet_cond) * 100,
                     "text_cond": np.nanmean(s.text_cond) * 100})
    RA = pd.DataFrame(resA); RA.to_csv(out / "d_resolution.csv", index=False)
    nat, low = RA.iloc[0], RA.iloc[-1]
    for col, lab in (("osnet", "OSNet"), ("irra", "IRRA"), ("text", "description")):
        say(f"   retained at 16 px: {lab:<12}{low[col] / nat[col] * 100:5.1f}%")

    # ---------------- B
    say("\n" + "=" * 100)
    say("B  PER-CANDIDATE QUALITY WEIGHTING vs TUNED CONSTANT (mixed gallery, fitted on validation)")
    say("=" * 100)
    resB = []
    for br, ac in (("osnet", a_os_m), ("irra", a_ir_m)):
        r = per_candidate(mixed, lvl, GP, GC, val, test, br, ac)
        say(f"  {br:<6} fitted centre {r['centre']:.2f} slope {r['slope']:+.2f} | "
            f"rule {r['rule']:.2f} vs constant {r['const']:.2f} (a={ac:.2f}) -> {fmt(*r['diff'][:3])}")
        resB.append({k: v for k, v in r.items() if k != "diff"} |
                    {"diff": r["diff"][0], "lo": r["diff"][1], "hi": r["diff"][2]})
    pd.DataFrame(resB).to_csv(out / "d_per_candidate.csv", index=False)

    # ---------------- C + D
    T, Vd = exploratory(gals["native"], GP, GC, gcams, test, val, a_os_n, a_ir_n)
    T.to_csv(out / "d_exploratory_perquery.csv", index=False)
    say("\n" + "=" * 100)
    say("C  EXPLORATORY ANALYSES (held-out identities, native gallery, OSNet branch)")
    say("=" * 100)
    m = T[(T.hit_img > 0) | (T.hit_txt > 0)]
    jac = (m.overlap / m.union.replace(0, np.nan)).mean()
    say(f"  correct results in top 10: photo {m.hit_img.mean():.2f} | description {m.hit_txt.mean():.2f} "
        f"| both {m.overlap.mean():.2f} | union {m.union.mean():.2f}")
    say(f"  overlap / union {jac:.3f} | description finds missed by photo "
        f"{(m.hit_txt - m.overlap).sum() / m.hit_txt.sum() * 100:.0f}% | photo finds missed by description "
        f"{(m.hit_img - m.overlap).sum() / m.hit_img.sum() * 100:.0f}%")
    d = T.dropna(subset=["ap_txt1", "ap_txt2"])
    dt = (d.ap_txt1 - d.ap_txt2).abs() * 100; df_ = (d.ap_fus1 - d.ap_fus2).abs() * 100
    red = paired(dt.values / 100, df_.values / 100, d.pid.values)
    say(f"  two independently written descriptions: mean |difference| description alone {dt.mean():.2f}, "
        f"fused {df_.mean():.2f}; reduction {fmt(*red[:3])}")
    one_q = d.drop_duplicates("pid")
    say(f"  searches whose two descriptions differ by >40 points: {(dt > 40).mean() * 100:.1f}% of searches "
        f"({(dt > 40).sum()} of {len(dt)})")
    g = paired(T.ap_txt_avg, T.ap_txt1, T.pid)
    say(f"  averaging two descriptions vs the first: {fmt(*g[:3])}  "
        f"({T.ap_txt1.mean() * 100:.2f} -> {T.ap_txt_avg.mean() * 100:.2f})")
    b = pd.cut(T.cap_len, [0, 18, 24, 30, 10 ** 6], labels=["<18", "18-24", "24-30", ">30"])
    say("  description length (words) vs description-only mAP: " +
        ", ".join(f"{l} {T.ap_txt1[b == l].mean() * 100:.1f} (n={int((b == l).sum())})" for l in b.cat.categories))
    cc = T[T.cams_true > 0]
    say(f"  camera coverage vs ground truth (top 20, complete system): {cc.cams_found.sum()}/{cc.cams_true.sum()} "
        f"= {cc.cams_found.sum() / cc.cams_true.sum() * 100:.1f}%  (mean {cc.cams_true.mean():.2f} cameras per person)")

    say("\n" + "=" * 100)
    say("D  CONFIDENCE - quintile thresholds fixed on VALIDATION, applied to held-out identities")
    say("=" * 100)
    edges = np.quantile(Vd.conf, [0.2, 0.4, 0.6, 0.8])
    band = np.digitize(T.conf, edges)
    parts = []
    for k, lab in enumerate(["lowest", "second", "middle", "fourth", "highest"]):
        s = T[band == k]
        mm = cboot(s.r1.values * 100, s.pid.values) if len(s) else (np.nan,) * 3
        parts.append(f"{lab} {mm[0]:.0f}% [{mm[1]:.0f},{mm[2]:.0f}] n={len(s)}")
    say("  Rank-1 by confidence band: " + " | ".join(parts))
    say(f"  overall Rank-1 {T.r1.mean() * 100:.1f}%")

    (out / "d_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    say(f"\nsaved to {out}   [{(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
