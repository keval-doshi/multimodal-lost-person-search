"""
Phase 8 - The complete system.

Every gain so far has been measured against a different baseline in a
different configuration. This runs everything together and answers the
question a reader will actually ask: what does the finished system achieve,
and how much does each part contribute once the others are present?

  python phase8_full.py --root .\\dataset

Two tables are produced.

  PROGRESSIVE BUILD-UP   components added one at a time, showing the path
                         from a photograph alone to the complete system.

  LEAVE-ONE-OUT          each component removed from the complete system in
                         turn. This is the honest measure of contribution:
                         a component that looks strong alone may add nothing
                         once the others are present, because their gains
                         overlap.

Components
  fusion              photograph and description combined, alpha = 0.45
  averaged captions   the two descriptions of a person averaged before use
  averaged photos     all available query photographs of a person averaged
  query expansion     query re-formed from its own top results
  k-reciprocal        neighbourhood-based re-ranking
  resolution matching query encoded at each candidate's own resolution

Protocol note on averaged photographs
-------------------------------------
PRW supplies up to one query box per camera per identity. Averaging them
means several cameras contribute to the query, so every camera that
contributed is excluded from the gallery for that search. The single-photo
baseline is scored against the SAME reduced gallery, otherwise the comparison
would be unfair.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

ALPHA = 0.45
LEVELS = ["native", "96px", "64px", "48px", "32px", "24px", "16px"]
LEVEL_PX = {"native": 10 ** 6, "96px": 96, "64px": 64, "48px": 48,
            "32px": 32, "24px": 24, "16px": 16}
BOUNDS = [20, 28, 40, 56, 80, 128]
ORDER = ["16px", "24px", "32px", "48px", "64px", "96px", "native"]


def zrow(v):
    return (v - v.mean()) / (v.std() + 1e-8)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def ap_of(scores, rel):
    n = int(rel.sum())
    if n == 0:
        return np.nan
    o = np.argsort(-scores, kind="stable")
    ro = rel[o]
    pos = np.flatnonzero(ro)
    return float(np.mean((np.arange(n) + 1) / (pos + 1)))


def r1_of(scores, rel):
    if rel.sum() == 0:
        return np.nan
    return bool(rel[np.argmax(scores)])


def ci(v, reps=4000, seed=0):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), size=(reps, len(v)))].mean(axis=1)
    return v.mean(), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def query_expansion(q, g, s, k=5, power=3.0):
    top = np.argsort(-s)[:k]
    w = np.clip(s[top], 0, None) ** power
    if w.sum() < 1e-9:
        return q
    return unit(q + (g[top] * w[:, None]).sum(axis=0) / w.sum())


def k_reciprocal(sub_sim, q_sim, k1=20, k2=6, lam=0.3):
    N = len(q_sim)
    if N < 4:
        return q_sim
    k1 = min(k1, N - 1)
    k2 = max(1, min(k2, k1))
    full = np.zeros((N + 1, N + 1), np.float32)
    full[0, 1:] = q_sim
    full[1:, 0] = q_sim
    full[1:, 1:] = sub_sim
    np.fill_diagonal(full, 1.0)
    order = np.argsort(-full, axis=1)

    def rec(i, k):
        fwd = order[i, :k + 1]
        return set(j for j in fwd if i in order[j, :k + 1]) | {i}

    V = np.zeros((N + 1, N + 1), np.float32)
    for i in range(N + 1):
        R = rec(i, k1)
        Rs = set(R)
        for j in list(R):
            Rj = rec(j, int(round(k1 / 2)))
            if len(R & Rj) >= (2.0 / 3.0) * len(Rj):
                Rs |= Rj
        idx = np.fromiter(Rs, int)
        w = np.exp(full[i, idx] - full[i, idx].max())
        V[i, idx] = w / (w.sum() + 1e-12)
    if k2 > 1:
        Vq = np.zeros_like(V)
        for i in range(N + 1):
            Vq[i] = V[order[i, :k2]].mean(axis=0)
        V = Vq
    vq = V[0]
    inter = np.minimum(vq[None, :], V[1:]).sum(axis=1)
    union = np.maximum(vq[None, :], V[1:]).sum(axis=1)
    return (1 - lam) * (inter / (union + 1e-12)) + lam * q_sim


def main():
    P = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_argument("--root", default="dataset")
    P.add_argument("--alpha", type=float, default=ALPHA)
    P.add_argument("--topn", type=int, default=200)
    P.add_argument("--qe-k", type=int, default=8)
    P.add_argument("--kr", type=float, nargs=3, default=[20, 6, 0.3],
                   metavar=("K1", "K2", "LAM"))
    P.add_argument("--gallery", default="both",
                   choices=["native", "mixed", "both"])
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--limit", type=int, default=0)
    a_ = P.parse_args()

    root = Path(a_.root).resolve()
    cache = root.parent / "cache"
    from phase3_baseline import load_split

    G = np.load(cache / "gallery.npz", allow_pickle=True)
    Q = np.load(cache / "query.npz", allow_pickle=True)
    C = np.load(cache / "captions.npz", allow_pickle=True)
    deg = cache / "degraded"
    qd_path = cache / "query_degraded.npz"
    QD = np.load(qd_path, allow_pickle=True) if qd_path.exists() else None
    if QD is None:
        print("(query_degraded.npz missing - resolution matching disabled)\n")

    sp = load_split(root / "PRW", "frame_test.mat")
    mask = np.array([f in sp for f in G["frames"]]) if sp else np.ones(len(G["frames"]), bool)
    g_pid_n, g_cam_n = G["ids"][mask], G["cams"][mask]
    g_h = G["height"][mask]
    g_os_n = G["osnet"][mask].astype(np.float32)
    g_ir_n = G["irra"][mask].astype(np.float32)

    q_os, q_ir = Q["osnet"].astype(np.float32), Q["irra"].astype(np.float32)
    q_pid, q_cam, q_frames = Q["ids"], Q["cams"], Q["frames"]
    c_ir = C["irra"].astype(np.float32)

    key = {(int(p), str(f)): i for i, (p, f) in enumerate(zip(q_pid, q_frames))}
    caps_of = {}
    for i, (p, f) in enumerate(zip(C["ids"], C["frames"])):
        r = key.get((int(p), str(f)))
        if r is not None:
            caps_of.setdefault(r, []).append(i)

    # queries grouped by identity, for the averaged-photograph variant
    by_id = {}
    for i in range(len(q_pid)):
        if i in caps_of:
            by_id.setdefault(int(q_pid[i]), []).append(i)

    qs = [i for i in range(len(q_pid)) if i in caps_of]
    ids = np.unique(q_pid[qs])
    rng = np.random.default_rng(a_.seed)
    sh = ids.copy(); rng.shuffle(sh)
    test_ids = set(sh[len(sh) // 2:].tolist())
    tq = [i for i in qs if int(q_pid[i]) in test_ids]
    if a_.limit:
        tq = tq[:a_.limit]
    print(f"Held-out: {len(tq)} queries, {len(test_ids)} identities")
    n_multi = sum(1 for i in tq if len(by_id[int(q_pid[i])]) > 1)
    print(f"  {n_multi} have a second query photograph available "
          f"(up to {max(len(by_id[int(q_pid[i])]) for i in tq)} exist per person, "
          f"at most 2 are used)\n")

    q_os_lv = ({lv: QD[f"os_{lv}"].astype(np.float32) for lv in LEVELS}
               if QD is not None else None)

    # ---------------- gallery configurations
    galleries = {}
    if a_.gallery in ("native", "both"):
        lvl = np.array([ORDER[int(np.digitize(h, BOUNDS))] for h in g_h])
        galleries["native"] = (g_os_n, g_ir_n, g_pid_n, g_cam_n, lvl)
    if a_.gallery in ("mixed", "both") and (deg / "meta.npz").exists():
        M = np.load(deg / "meta.npz")
        lf = {lv: np.load(deg / f"gallery_{lv}.npz") for lv in LEVELS}
        n_d = len(M["ids"])
        pick = np.random.default_rng(a_.seed).integers(0, len(LEVELS), size=n_d)
        gm = np.zeros((n_d, 512), np.float32); gmi = np.zeros((n_d, 512), np.float32)
        lvm = np.empty(n_d, object)
        for k, lv in enumerate(LEVELS):
            m = pick == k
            if m.any():
                gm[m] = lf[lv]["osnet"][m].astype(np.float32)
                gmi[m] = lf[lv]["irra"][m].astype(np.float32)
                lvm[m] = lv
        galleries["mixed"] = (gm, gmi, M["ids"], M["cams"], lvm.astype(str))

    ALL = ["fusion", "avg_captions", "avg_photos", "qe", "krecip", "resmatch"]
    rows = []

    for gname, (g_img, g_txt, gp, gc, g_lvl) in galleries.items():
        print("=" * 78)
        print(f"GALLERY: {gname}")
        print("=" * 78)

        def run(components, collect_r1=False):
            aps, r1s = [], []
            for qi in tq:
                pid = int(q_pid[qi])
                # At most two photographs are averaged. PRW supplies one query
                # box per camera per identity, so averaging all of them would
                # mean excluding up to six cameras and leaving no gallery at
                # all. Two keeps at least four cameras searchable.
                group = [qi] + [j for j in by_id[pid] if j != qi][:1]
                use_multi = "avg_photos" in components and len(group) > 1
                # Every camera contributing a photograph is excluded, and the
                # SAME exclusion is applied whether or not averaging is used,
                # so all variants are scored against an identical gallery.
                excl = set(q_cam[group].tolist())
                idx = np.flatnonzero(~np.isin(gc, list(excl)))
                rel = (gp[idx] == pid) if len(idx) else np.zeros(0, bool)
                if not rel.any():
                    continue
                gi, gt = g_img[idx], g_txt[idx]

                qv = unit(q_os[group].mean(axis=0)) if use_multi else q_os[qi]

                if "resmatch" in components and q_os_lv is not None:
                    # The averaged query also has a version at every level:
                    # the mean of the group's encodings at that level. Earlier
                    # this branch was skipped when averaging was active, which
                    # silently disabled resolution matching in the complete
                    # system and made its contribution read as exactly zero.
                    lv_idx = g_lvl[idx]
                    z_img = np.empty(len(idx), np.float32)
                    for lv in LEVELS:
                        m = (lv_idx == lv)
                        if not m.any():
                            continue
                        qlv = (unit(q_os_lv[lv][group].mean(axis=0))
                               if use_multi else q_os_lv[lv][qi])
                        z_img[m] = zrow(qlv @ gi.T)[m]
                else:
                    z_img = zrow(qv @ gi.T)

                cl = caps_of[qi]
                if "avg_captions" in components and len(cl) > 1:
                    tvec = unit(c_ir[cl[:2]].mean(axis=0))
                else:
                    tvec = c_ir[cl[0]]
                z_txt = zrow(tvec @ gt.T)

                s = (a_.alpha * z_img + (1 - a_.alpha) * z_txt
                     if "fusion" in components else z_img)

                if "qe" in components:
                    qv2 = query_expansion(qv, gi, s, k=a_.qe_k)
                    z2 = zrow(qv2 @ gi.T)
                    s = (a_.alpha * z2 + (1 - a_.alpha) * z_txt
                         if "fusion" in components else z2)

                if "krecip" in components:
                    n = min(a_.topn, len(s))
                    top = np.argsort(-s)[:n]
                    sub = gi[top]
                    s2 = np.full(len(s), -1e9, np.float32)
                    s2[top] = k_reciprocal(sub @ sub.T, s[top],
                                           int(a_.kr[0]), int(a_.kr[1]), a_.kr[2])
                    s = s2

                aps.append(ap_of(s, rel))
                if collect_r1:
                    r1s.append(r1_of(s, rel))
            if not aps:
                print("  WARNING: no query produced a scorable result. The "
                      "gallery is probably empty after camera exclusion.")
            return np.array(aps), (np.array(r1s, float) if collect_r1 else None)

        # -------- progressive build-up
        print("\nPROGRESSIVE BUILD-UP")
        print(f"  {'system':<34}{'mAP':>8}{'95% CI':>18}{'gain':>9}")
        print("  " + "-" * 70)
        steps = [("photograph only", []),
                 ("+ description (fusion)", ["fusion"]),
                 ("+ averaged captions", ["fusion", "avg_captions"]),
                 ("+ averaged photographs", ["fusion", "avg_captions", "avg_photos"]),
                 ("+ query expansion", ["fusion", "avg_captions", "avg_photos", "qe"]),
                 ("+ k-reciprocal", ["fusion", "avg_captions", "avg_photos", "qe", "krecip"])]
        if q_os_lv is not None:
            steps.append(("+ resolution matching", ALL))

        prev = None
        t0 = time.time()
        full_ap = None
        probe, _ = run([])
        n_ok = int((~np.isnan(probe)).sum())
        print(f"  {n_ok} of {len(tq)} queries scorable after camera exclusion")
        if n_ok == 0:
            print("  ABORTING this gallery: nothing to score.")
            continue
        for lab, comp in steps:
            ap, _ = run(comp)
            m, lo, hi = ci(ap * 100)
            g = "" if prev is None else f"{m - prev:+8.2f}"
            print(f"  {lab:<34}{m:>8.2f}  [{lo:5.2f}, {hi:5.2f}]{g}")
            rows.append({"gallery": gname, "stage": lab, "mAP": round(m, 2),
                         "lo": round(lo, 2), "hi": round(hi, 2)})
            prev = m
            full_ap = ap
        base_ap, _ = run([])
        d, dl, dh = ci((full_ap - base_ap) * 100)
        print(f"\n  complete system over photograph alone: "
              f"{d:+.2f}  [{dl:+.2f}, {dh:+.2f}]")
        print(f"  ({(d / max(base_ap.mean() * 100, 1e-9)) * 100:.0f}% relative)")
        print(f"  [{time.time() - t0:.0f}s]")

        # -------- leave-one-out
        print("\nLEAVE-ONE-OUT FROM THE COMPLETE SYSTEM")
        print("  A component that looks strong alone may add nothing once the")
        print("  others are present, because their gains overlap.\n")
        print(f"  {'component removed':<26}{'mAP':>8}{'change':>10}{'95% CI':>20}")
        print("  " + "-" * 66)
        full_m = full_ap[~np.isnan(full_ap)].mean() * 100
        present = [c for c in ALL if q_os_lv is not None or c != "resmatch"]
        for c in present:
            sub = [x for x in present if x != c]
            ap, _ = run(sub)
            n = min(len(ap), len(full_ap))
            d, dl, dh = ci((ap[:n] - full_ap[:n]) * 100)
            sig = "" if (dl < 0 < dh) else "  significant"
            print(f"  {'without ' + c:<26}{ap[~np.isnan(ap)].mean() * 100:>8.2f}"
                  f"{d:>10.2f}   [{dl:+.2f}, {dh:+.2f}]{sig}")
            rows.append({"gallery": gname, "stage": f"without {c}",
                         "mAP": round(ap[~np.isnan(ap)].mean() * 100, 2),
                         "lo": round(dl, 2), "hi": round(dh, 2)})
        print(f"\n  complete system: {full_m:.2f} mAP")
        print("  A large negative change means the component is doing real work.")
        print()

    out = cache / "phase8_full.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
