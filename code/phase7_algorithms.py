"""
Phase 7a - Retrieval algorithms.

Four additions that change how retrieval works, rather than only measuring
it. All run from cached features, so no GPU is required.

  python phase7_algorithms.py --root .\\dataset

A1  K-RECIPROCAL RE-RANKING          (Zhong et al., CVPR 2017)
    The standard post-processing step in person re-identification. Two
    people are more likely the same if they are in each other's near
    neighbourhoods, not merely if one is near the other. The ranking is
    refined using a Jaccard distance over expanded reciprocal neighbour
    sets.

    Adaptation made here: the initial ranking is the FUSED photograph-plus-
    description score, but the neighbourhood structure has to come from
    appearance alone, because gallery crops have no descriptions attached.
    Mixing the two sources this way is not in the original method and is a
    design decision specific to the multimodal setting.

A3  QUERY EXPANSION
    Average the query with its own top-k results and search again. Standard
    in image retrieval; cheap and usually worth a few points.

A4  CROSS-MODAL RANK AGREEMENT
    A candidate ranked highly by BOTH branches is more trustworthy than one
    ranked first by a single branch. Weighted fusion cannot express this,
    because it is additive: a very high image score alone can carry a
    candidate. Scoring on the WORSE of the two ranks makes the signal
    conjunctive instead.

T1  WRONG-DESCRIPTION ROBUSTNESS
    Witnesses misremember. Every result so far assumes the description is
    accurate. This pairs each photograph with another person's description
    and measures the damage.

Every hyperparameter is tuned on an identity-disjoint validation half and
reported on the held-out half.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

ALPHA = 0.45
TOPN = 200          # candidates carried into re-ranking


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


def ci(v, reps=3000, seed=0):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), size=(reps, len(v)))].mean(axis=1)
    return v.mean(), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ------------------------------------------------------------------ A1
def k_reciprocal(sub_sim, q_sim, k1=20, k2=6, lam=0.3):
    """
    K-reciprocal re-ranking restricted to a candidate shortlist.

    sub_sim : (N,N) gallery-gallery similarity among the shortlist
    q_sim   : (N,)  original query-to-candidate score, higher is better
    Returns a refined score, higher is better.

    The full method builds neighbour sets over the entire gallery. Running
    it on the top N candidates is the usual practical approximation and
    keeps the gallery-gallery matrix to N by N rather than 27,778 squared.
    """
    N = len(q_sim)
    k1 = min(k1, N - 1)
    k2 = max(1, min(k2, k1))
    if N < 4:
        return q_sim

    # query treated as another node so its neighbourhood is comparable
    full = np.zeros((N + 1, N + 1), np.float32)
    full[0, 1:] = q_sim
    full[1:, 0] = q_sim
    full[1:, 1:] = sub_sim
    np.fill_diagonal(full, 1.0)

    order = np.argsort(-full, axis=1)

    def reciprocal_set(i, k):
        fwd = order[i, :k + 1]
        out = [j for j in fwd if i in order[j, :k + 1]]
        return set(out) | {i}

    V = np.zeros((N + 1, N + 1), np.float32)
    for i in range(N + 1):
        R = reciprocal_set(i, k1)
        # expansion: add a neighbour's own set when the two overlap heavily
        Rstar = set(R)
        for j in list(R):
            Rj = reciprocal_set(j, int(np.round(k1 / 2)))
            if len(R & Rj) >= (2.0 / 3.0) * len(Rj):
                Rstar |= Rj
        idx = np.fromiter(Rstar, int)
        w = np.exp(full[i, idx] - full[i, idx].max())
        V[i, idx] = w / (w.sum() + 1e-12)

    # local query expansion over the k2 nearest neighbours
    if k2 > 1:
        Vq = np.zeros_like(V)
        for i in range(N + 1):
            nb = order[i, :k2]
            Vq[i] = V[nb].mean(axis=0)
        V = Vq

    # Jaccard similarity between the query's set and each candidate's
    vq = V[0]
    inter = np.minimum(vq[None, :], V[1:]).sum(axis=1)
    union = np.maximum(vq[None, :], V[1:]).sum(axis=1)
    jac = inter / (union + 1e-12)

    return (1 - lam) * jac + lam * zrow(q_sim) * 0.1 + lam * q_sim


# ------------------------------------------------------------------ A3
def query_expansion(q_vec, g_vecs, scores, k=5, power=3.0):
    """Average the query with its top-k results, weighted by score."""
    top = np.argsort(-scores)[:k]
    w = np.clip(scores[top], 0, None) ** power
    if w.sum() < 1e-9:
        return q_vec
    new = q_vec + (g_vecs[top] * w[:, None]).sum(axis=0) / w.sum()
    return new / (np.linalg.norm(new) + 1e-8)


# ------------------------------------------------------------------ A4
def rank_agreement(z_img, z_txt, k0=30.0, mode="conjunctive"):
    """
    Rank-based combination, in two opposite flavours.

    conjunctive - score on the WORSE of the two ranks, so a candidate must
                  satisfy BOTH branches. Filters noise when the two sources
                  are redundant.
    disjunctive - score on the BETTER of the two ranks, so a candidate needs
                  only ONE branch to find it. Preserves coverage when the
                  two sources are complementary.

    Which should win is predictable in advance. The Phase 6 analysis measured
    a Jaccard overlap of 0.195 between the correct sets of the two branches,
    meaning they find largely different people. Under that condition the
    conjunctive form should DESTROY the complementary signal and the
    disjunctive form should preserve it. This experiment tests that
    prediction rather than merely trying both.
    """
    ri = np.empty(len(z_img)); ri[np.argsort(-z_img)] = np.arange(len(z_img))
    rt = np.empty(len(z_txt)); rt[np.argsort(-z_txt)] = np.arange(len(z_txt))
    r = np.maximum(ri, rt) if mode == "conjunctive" else np.minimum(ri, rt)
    return 1.0 / (k0 + r)


def main():
    P = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_argument("--root", default="dataset")
    P.add_argument("--alpha", type=float, default=ALPHA)
    P.add_argument("--topn", type=int, default=TOPN)
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--limit", type=int, default=0, help="smoke test on N queries")
    a_ = P.parse_args()

    root = Path(a_.root).resolve()
    cache = root.parent / "cache"
    from phase3_baseline import load_split

    G = np.load(cache / "gallery.npz", allow_pickle=True)
    Q = np.load(cache / "query.npz", allow_pickle=True)
    C = np.load(cache / "captions.npz", allow_pickle=True)

    sp = load_split(root / "PRW", "frame_test.mat")
    mask = np.array([f in sp for f in G["frames"]]) if sp else np.ones(len(G["frames"]), bool)
    g_os = G["osnet"][mask].astype(np.float32)
    g_ir = G["irra"][mask].astype(np.float32)
    g_pid, g_cam = G["ids"][mask], G["cams"][mask]

    q_os, q_ir = Q["osnet"].astype(np.float32), Q["irra"].astype(np.float32)
    q_pid, q_cam, q_frames = Q["ids"], Q["cams"], Q["frames"]

    key = {(int(p), str(f)): i for i, (p, f) in enumerate(zip(q_pid, q_frames))}
    caps_of = {}
    for i, (p, f) in enumerate(zip(C["ids"], C["frames"])):
        r = key.get((int(p), str(f)))
        if r is not None:
            caps_of.setdefault(r, []).append(i)
    c_ir = C["irra"].astype(np.float32)

    valid_q = [i for i in range(len(q_pid)) if i in caps_of]
    if a_.limit:
        valid_q = valid_q[:a_.limit]
    print(f"Gallery {len(g_pid)} | usable queries {len(valid_q)}")

    ids = np.unique(q_pid[valid_q])
    rng = np.random.default_rng(a_.seed)
    rng.shuffle(ids)
    val_ids = set(ids[:len(ids) // 2].tolist())
    vq = [i for i in valid_q if int(q_pid[i]) in val_ids]
    tq = [i for i in valid_q if int(q_pid[i]) not in val_ids]
    print(f"Identity-disjoint split: {len(vq)} validation / {len(tq)} test queries\n")

    def score_query(qi, ci_idx=None):
        """Return the ingredients needed by every method for one query."""
        keep = (g_cam != q_cam[qi])
        idx = np.flatnonzero(keep)
        rel = (g_pid[idx] == q_pid[qi])
        zi = zrow(q_os[qi] @ g_os[idx].T)
        c = ci_idx if ci_idx is not None else caps_of[qi][0]
        zt = zrow(c_ir[c] @ g_ir[idx].T)
        return idx, rel, zi, zt

    # -------------------------------------------------- tune on validation
    print("Tuning hyperparameters on the validation half ...")

    def eval_set(qs, method, **kw):
        out = []
        for qi in qs:
            idx, rel, zi, zt = score_query(qi)
            fused = a_.alpha * zi + (1 - a_.alpha) * zt
            if method == "baseline":
                s = fused
            elif method == "qe":
                k = kw.get("k", 5)
                if k == 0:
                    s = fused
                else:
                    qv = query_expansion(q_os[qi], g_os[idx], fused, k=k)
                    s = (a_.alpha * zrow(qv @ g_os[idx].T)
                         + (1 - a_.alpha) * zt)
            elif method == "agree":
                lam = kw.get("lam", 1.0)
                s = fused if lam == 0 else fused + lam * zrow(
                    rank_agreement(zi, zt, mode=kw.get("mode", "conjunctive")))
            elif method == "krecip":
                n = min(a_.topn, len(fused))
                top = np.argsort(-fused)[:n]
                sub = g_os[idx][top]
                s2 = k_reciprocal(sub @ sub.T, fused[top],
                                  k1=kw.get("k1", 20), k2=kw.get("k2", 6),
                                  lam=kw.get("lam", 0.3))
                s = np.full(len(fused), -1e9, np.float32)
                s[top] = s2
            else:
                raise ValueError(method)
            out.append(ap_of(s, rel))
        return np.nanmean(out) * 100

    base_val = eval_set(vq, "baseline")
    print(f"  baseline                       {base_val:.2f}")

    # every grid contains the do-nothing option, so a harmful component can
    # switch itself off rather than being forced on. Phase 5 showed what
    # happens when a search space does not contain its own baseline.
    best_qe = max([0, 3, 5, 8, 12], key=lambda k: eval_set(vq, "qe", k=k))
    print(f"  query expansion  k={best_qe:<3}            "
          f"{eval_set(vq, 'qe', k=best_qe):.2f}"
          f"{'   (switched off)' if best_qe == 0 else ''}")

    cand = [(l, mo) for mo in ("conjunctive", "disjunctive")
            for l in (0.0, 0.5, 1.0, 2.0)]
    print("\n  rank agreement, both directions")
    for mo in ("conjunctive", "disjunctive"):
        for l in (0.5, 1.0, 2.0):
            print(f"    {mo:<12} lambda={l:<4} "
                  f"{eval_set(vq, 'agree', lam=l, mode=mo):.2f}")
    best_lam, best_mode = max(cand,
                              key=lambda c: eval_set(vq, "agree",
                                                     lam=c[0], mode=c[1]))
    print(f"  chosen: {best_mode} lambda={best_lam}"
          f"{'   (switched off)' if best_lam == 0 else ''}")

    grid = [(20, 6, 0.3), (20, 6, 0.5), (30, 6, 0.3), (10, 3, 0.3)]
    best_kr = max(grid, key=lambda g: eval_set(vq[:200], "krecip",
                                               k1=g[0], k2=g[1], lam=g[2]))
    print(f"  k-reciprocal     k1={best_kr[0]} k2={best_kr[1]} "
          f"lam={best_kr[2]}  "
          f"{eval_set(vq[:200], 'krecip', k1=best_kr[0], k2=best_kr[1], lam=best_kr[2]):.2f}"
          f"   (tuned on 200 queries for speed)")

    # -------------------------------------------------- held-out
    print(f"\n{'=' * 72}")
    print("HELD-OUT RESULTS  (mAP, 95% bootstrap interval)")
    print("=" * 72)

    per_q = {k: [] for k in ("image only", "text only", "baseline fusion",
                             "+ query expansion", "+ rank agreement",
                             "conjunctive (forced)", "disjunctive (forced)",
                             "+ k-reciprocal", "+ all three")}
    for n, qi in enumerate(tq, 1):
        idx, rel, zi, zt = score_query(qi)
        fused = a_.alpha * zi + (1 - a_.alpha) * zt

        per_q["image only"].append(ap_of(zi, rel))
        per_q["text only"].append(ap_of(zt, rel))
        per_q["baseline fusion"].append(ap_of(fused, rel))

        qv = query_expansion(q_os[qi], g_os[idx], fused, k=best_qe)
        s_qe = a_.alpha * zrow(qv @ g_os[idx].T) + (1 - a_.alpha) * zt
        per_q["+ query expansion"].append(ap_of(s_qe, rel))

        s_ag = (fused if best_lam == 0 else
                fused + best_lam * zrow(rank_agreement(zi, zt, mode=best_mode)))
        per_q["+ rank agreement"].append(ap_of(s_ag, rel))
        s_conj = fused + 1.0 * zrow(rank_agreement(zi, zt, mode="conjunctive"))
        s_disj = fused + 1.0 * zrow(rank_agreement(zi, zt, mode="disjunctive"))
        per_q["conjunctive (forced)"].append(ap_of(s_conj, rel))
        per_q["disjunctive (forced)"].append(ap_of(s_disj, rel))

        nn = min(a_.topn, len(fused))
        top = np.argsort(-fused)[:nn]
        sub = g_os[idx][top]
        s_kr = np.full(len(fused), -1e9, np.float32)
        s_kr[top] = k_reciprocal(sub @ sub.T, fused[top], *best_kr)
        per_q["+ k-reciprocal"].append(ap_of(s_kr, rel))

        comb = (s_qe if best_lam == 0 else
                s_qe + best_lam * zrow(rank_agreement(zi, zt, mode=best_mode)))
        top2 = np.argsort(-comb)[:nn]
        sub2 = g_os[idx][top2]
        s_all = np.full(len(comb), -1e9, np.float32)
        s_all[top2] = k_reciprocal(sub2 @ sub2.T, comb[top2], *best_kr)
        per_q["+ all three"].append(ap_of(s_all, rel))

        if n % 200 == 0:
            print(f"  {n}/{len(tq)} queries")

    base = np.array(per_q["baseline fusion"])
    print()
    for k, v in per_q.items():
        v = np.array(v)
        m, lo, hi = ci(v * 100)
        if k in ("image only", "text only", "baseline fusion"):
            print(f"  {k:<20}{m:6.2f}  [{lo:5.2f}, {hi:5.2f}]")
        else:
            d, dlo, dhi = ci((v - base) * 100)
            sig = "significant" if (dlo > 0 or dhi < 0) else "not significant"
            print(f"  {k:<20}{m:6.2f}  [{lo:5.2f}, {hi:5.2f}]   "
                  f"{d:+5.2f} [{dlo:+.2f}, {dhi:+.2f}]  {sig}")

    # -------------------------------------------------- T1
    print(f"\n{'=' * 72}")
    print("T1.  WHAT IF THE DESCRIPTION IS WRONG?")
    print("=" * 72)
    print("Each photograph is paired with another person's description.")
    print("Witnesses misremember, so this bounds the damage.\n")

    rng2 = np.random.default_rng(1)
    wrong = {"image only": [], "correct description": [],
             "wrong description": [], "text only, wrong": []}
    for qi in tq:
        idx, rel, zi, zt = score_query(qi)
        others = [j for j in range(len(c_ir))
                  if int(C["ids"][j]) != int(q_pid[qi])]
        zw = zrow(c_ir[rng2.choice(others)] @ g_ir[idx].T)
        wrong["image only"].append(ap_of(zi, rel))
        wrong["correct description"].append(
            ap_of(a_.alpha * zi + (1 - a_.alpha) * zt, rel))
        wrong["wrong description"].append(
            ap_of(a_.alpha * zi + (1 - a_.alpha) * zw, rel))
        wrong["text only, wrong"].append(ap_of(zw, rel))

    img = np.array(wrong["image only"])
    for k, v in wrong.items():
        v = np.array(v)
        m, lo, hi = ci(v * 100)
        extra = ""
        if k.startswith("wrong"):
            d, dlo, dhi = ci((v - img) * 100)
            extra = f"   vs photograph alone {d:+.2f} [{dlo:+.2f}, {dhi:+.2f}]"
        print(f"  {k:<22}{m:6.2f}  [{lo:5.2f}, {hi:5.2f}]{extra}")

    dmg = np.array(wrong["wrong description"]) - img
    print(f"\n  A wrong description moves accuracy by {dmg.mean() * 100:+.2f} "
          f"points relative to using the photograph alone.")
    if dmg.mean() < -0.01:
        print("  Fusion is therefore NOT safe when the description may be")
        print("  mistaken; the system should fall back to the photograph when")
        print("  confidence in the description is low.")
    else:
        print("  Fusion degrades gracefully: even a wrong description does not")
        print("  drag the result below the photograph alone, because the")
        print("  photograph anchors the ranking.")

    out = cache / "phase7_algorithms.csv"
    pd.DataFrame({k: pd.Series(v) for k, v in per_q.items()}).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
