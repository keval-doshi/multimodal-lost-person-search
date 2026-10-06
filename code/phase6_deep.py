"""
Phase 6d - Second-level analysis.

Five questions the first analysis opened up. All run from cached features,
so this needs no GPU and takes a few minutes.

  python phase6_deep.py --root .\\dataset

1  WHICH ATTRIBUTES ACTUALLY HELP
   The caption text has not been examined. Which garments, colours and
   accessories make a description work, and does mentioning a RARE colour
   help more than a common one? This is guidance for interviewing a witness,
   and it does not exist in the literature for this task.

2  ORACLE PER-QUERY BALANCE
   Phase 5 tested a per-CROP balance and found nothing, for a structural
   reason: a per-item weight breaks comparability between candidates. A
   per-QUERY balance does not have that problem, since every candidate for a
   given query is weighted identically. This measures the headroom, then
   asks whether the ideal balance is predictable from things observable at
   search time.

3  AVERAGING TWO DESCRIPTIONS
   Wording alone moves average precision by 18 points. Each person carries
   two independently written descriptions, so averaging them should recover
   part of that. If it does, the operational advice is to ask twice.

4  CAN THE SYSTEM SAY WHEN IT IS CONFIDENT
   A list of twenty results with no confidence signal is far less useful
   than one that flags which results to trust. Tested against signals
   available at search time only.

5  DOES THE FUSION METHOD MATTER
   Weighted z-scores against reciprocal rank fusion, the standard
   alternative in information retrieval, which uses only the ordering and is
   therefore immune to score-scale problems.
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

ALPHA = 0.45

COLOURS = ["black", "white", "red", "blue", "grey", "gray", "green",
           "yellow", "brown", "pink", "orange", "purple", "dark", "light"]
GARMENTS = ["shirt", "t-shirt", "tshirt", "jacket", "coat", "trousers",
            "pants", "jeans", "shorts", "skirt", "dress", "shoes",
            "sneakers", "boots", "sweater", "hoodie", "suit"]
ACCESSORIES = ["bag", "backpack", "handbag", "hat", "cap", "glasses",
               "umbrella", "phone"]


def zrow(v):
    return (v - v.mean()) / (v.std() + 1e-8)


def ap_of(scores, rel, valid):
    s, r = scores[valid], rel[valid]
    n = int(r.sum())
    if n == 0:
        return np.nan
    o = np.argsort(-s, kind="stable")
    ro = r[o]
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


def main():
    P = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_argument("--root", default="dataset")
    P.add_argument("--alpha", type=float, default=ALPHA)
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
    c_text = [str(t) for t in C["text"]]

    print(f"Gallery {len(g_pid)} | queries {len(q_pid)} | captions {len(c_ir)}")
    print("Scoring every query ... this is the slow part\n")

    GRID = np.round(np.arange(0.0, 1.001, 0.05), 2)
    rows = []
    for n_done, qi in enumerate(range(len(q_pid)), 1):
        ci_list = caps_of.get(qi)
        if not ci_list:
            continue
        rel = (g_pid == q_pid[qi])
        valid = (g_cam != q_cam[qi])
        if not (rel & valid).any():
            continue

        zi = zrow(q_os[qi] @ g_os.T)
        zt1 = zrow(c_ir[ci_list[0]] @ g_ir.T)

        # averaged description: centroid of the two embeddings, renormalised
        emb = c_ir[ci_list[:2]].mean(axis=0)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        zt_avg = zrow(emb @ g_ir.T)

        fused = a_.alpha * zi + (1 - a_.alpha) * zt1
        fused_avg = a_.alpha * zi + (1 - a_.alpha) * zt_avg

        # oracle per-query balance - uses the answer, so an upper bound only
        best_a, best_ap = a_.alpha, -1.0
        for al in GRID:
            v = ap_of(al * zi + (1 - al) * zt1, rel, valid)
            if v > best_ap:
                best_a, best_ap = float(al), v

        # reciprocal rank fusion: ordering only, ignores score scale
        ri = np.empty(len(zi)); ri[np.argsort(-zi)] = np.arange(len(zi))
        rt = np.empty(len(zt1)); rt[np.argsort(-zt1)] = np.arange(len(zt1))
        rrf = 1.0 / (60 + ri) + 1.0 / (60 + rt)

        # confidence signals, all observable at search time
        sf = np.sort(fused[valid])[::-1]
        top_idx = np.flatnonzero(valid)[np.argsort(-fused[valid])[0]]

        rows.append({
            "qi": qi, "pid": int(q_pid[qi]),
            "ap_img": ap_of(zi, rel, valid),
            "ap_txt": ap_of(zt1, rel, valid),
            "ap_txt_avg": ap_of(zt_avg, rel, valid),
            "ap_fused": ap_of(fused, rel, valid),
            "ap_fused_avg": ap_of(fused_avg, rel, valid),
            "ap_oracle": best_ap, "alpha_oracle": best_a,
            "ap_rrf": ap_of(rrf, rel, valid),
            "conf_top": float(sf[0]),
            "conf_gap12": float(sf[0] - sf[1]),
            "conf_gap110": float(sf[0] - sf[9]) if len(sf) > 9 else np.nan,
            "rank1_correct": bool(rel[top_idx]),
            "cap": c_text[ci_list[0]],
            "cap_len": len(c_text[ci_list[0]].split()),
        })
        if n_done % 400 == 0:
            print(f"  {n_done}/{len(q_pid)} queries")

    df = pd.DataFrame(rows)
    print(f"\nAnalysed {len(df)} queries\n")

    # ------------------------------------------------------------- 1
    print("=" * 76)
    print("1.  WHICH ATTRIBUTES MAKE A DESCRIPTION WORK")
    print("=" * 76)
    low = [c.lower() for c in df.cap]

    def report(terms, title):
        print(f"\n  {title}")
        print(f"    {'term':<14}{'n':>6}{'share':>8}{'text AP':>10}"
              f"{'without':>10}{'effect':>9}")
        print("    " + "-" * 58)
        out = []
        for w in terms:
            has = np.array([bool(re.search(rf"\b{re.escape(w)}\b", t)) for t in low])
            if has.sum() < 40:
                continue
            a1 = df.ap_txt[has].mean() * 100
            a0 = df.ap_txt[~has].mean() * 100
            out.append((w, int(has.sum()), has.mean() * 100, a1, a0, a1 - a0))
        for w, n, sh, a1, a0, d in sorted(out, key=lambda x: -x[5]):
            print(f"    {w:<14}{n:>6}{sh:>7.1f}%{a1:>9.1f}%{a0:>9.1f}%{d:>+8.1f}")
        return out

    col = report(COLOURS, "colour terms")
    report(GARMENTS, "garments")
    report(ACCESSORIES, "accessories")

    if len(col) >= 4:
        srt = sorted(col, key=lambda x: x[1])
        rare = [w for w, *_ in srt[:len(srt) // 2]]
        common = [w for w, *_ in srt[len(srt) // 2:]]
        has_r = np.array([any(re.search(rf"\b{w}\b", t) for w in rare) for t in low])
        has_c = np.array([any(re.search(rf"\b{w}\b", t) for w in common) for t in low])
        only_r, only_c = has_r & ~has_c, has_c & ~has_r
        m1, l1, h1 = ci(df.ap_txt[only_r].values * 100)
        m0, l0, h0 = ci(df.ap_txt[only_c].values * 100)
        print(f"\n  DOES A RARE COLOUR HELP MORE?")
        print(f"    rare colours only   ({int(only_r.sum()):>4} captions)  "
              f"{m1:5.1f}%  [{l1:.1f}, {h1:.1f}]")
        print(f"    common colours only ({int(only_c.sum()):>4} captions)  "
              f"{m0:5.1f}%  [{l0:.1f}, {h0:.1f}]")
        print(f"    difference {m1 - m0:+.1f} points")
        print(f"    rare:   {', '.join(rare)}")
        print(f"    common: {', '.join(common)}")

    n_col = np.array([sum(bool(re.search(rf"\b{re.escape(w)}\b", t))
                          for w in COLOURS) for t in low])
    print(f"\n  NUMBER OF DISTINCT COLOUR TERMS MENTIONED")
    print(f"    {'colours':<10}{'n':>7}{'text AP':>10}{'fused AP':>11}")
    print("    " + "-" * 40)
    for k in range(0, 5):
        s = df[n_col == k] if k < 4 else df[n_col >= 4]
        if len(s) >= 30:
            print(f"    {(str(k) if k < 4 else '4+'):<10}{len(s):>7}"
                  f"{s.ap_txt.mean() * 100:>9.1f}%{s.ap_fused.mean() * 100:>10.1f}%")

    # ------------------------------------------------------------- 2
    print("\n" + "=" * 76)
    print("2.  ORACLE PER-QUERY BALANCE - IS THERE HEADROOM?")
    print("=" * 76)
    m_f, l_f, h_f = ci(df.ap_fused.values * 100)
    m_o, l_o, h_o = ci(df.ap_oracle.values * 100)
    m_d, l_d, h_d = ci((df.ap_oracle - df.ap_fused).values * 100)
    print(f"    fixed balance {a_.alpha}      {m_f:6.2f}  [{l_f:.2f}, {h_f:.2f}]")
    print(f"    oracle per query        {m_o:6.2f}  [{l_o:.2f}, {h_o:.2f}]")
    print(f"    headroom                {m_d:+6.2f}  [{l_d:+.2f}, {h_d:+.2f}]")
    print(f"\n    The oracle uses the answer, so it is an upper bound rather")
    print(f"    than a method. It bounds what ANY per-query rule could gain.")
    print(f"\n    distribution of the ideal balance")
    b = pd.cut(df.alpha_oracle, [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
               labels=["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"])
    for lab in b.cat.categories:
        n = int((b == lab).sum())
        print(f"      {lab:<10}{n:>6}  ({n / len(df) * 100:5.1f}%)")
    print(f"\n    Is the ideal balance predictable from what we can see?")
    for c in ("ap_img", "ap_txt", "cap_len", "conf_gap12", "conf_top"):
        r = np.corrcoef(df[c].fillna(0), df.alpha_oracle)[0, 1]
        note = "   (not observable at search time)" if c in ("ap_img", "ap_txt") else ""
        print(f"      {c:<12}{r:+.3f}{note}")

    # ------------------------------------------------------------- 3
    print("\n" + "=" * 76)
    print("3.  DOES AVERAGING TWO DESCRIPTIONS HELP?")
    print("=" * 76)
    for lab, c_ in (("one description", "ap_txt"), ("two averaged", "ap_txt_avg")):
        m, lo, hi = ci(df[c_].values * 100)
        print(f"    text alone, {lab:<18}{m:6.2f}  [{lo:.2f}, {hi:.2f}]")
    m, lo, hi = ci((df.ap_txt_avg - df.ap_txt).values * 100)
    print(f"    gain                          {m:+6.2f}  [{lo:+.2f}, {hi:+.2f}]")
    for lab, c_ in (("one description", "ap_fused"), ("two averaged", "ap_fused_avg")):
        m2, lo2, hi2 = ci(df[c_].values * 100)
        print(f"    fused,      {lab:<18}{m2:6.2f}  [{lo2:.2f}, {hi2:.2f}]")
    m2, lo2, hi2 = ci((df.ap_fused_avg - df.ap_fused).values * 100)
    print(f"    gain                          {m2:+6.2f}  [{lo2:+.2f}, {hi2:+.2f}]")
    print(f"\n    Averaging costs nothing at search time and needs only a")
    print(f"    second description of the same person.")

    # ------------------------------------------------------------- 4
    print("\n" + "=" * 76)
    print("4.  CAN THE SYSTEM TELL WHEN IT IS CONFIDENT?")
    print("=" * 76)
    print(f"    overall rank-1 accuracy {df.rank1_correct.mean() * 100:.1f}%\n")
    for sig, lab in (("conf_gap12", "gap between ranks 1 and 2"),
                     ("conf_gap110", "gap between ranks 1 and 10"),
                     ("conf_top", "top score")):
        v = df[sig].fillna(0)
        q = pd.qcut(v, 5, labels=["lowest", "low", "mid", "high", "highest"],
                    duplicates="drop")
        accs = [df.rank1_correct[q == l].mean() * 100 for l in q.cat.categories]
        print(f"    {lab}")
        print("      " + "  ".join(f"{l}:{a:.0f}%" for l, a in
                                   zip(q.cat.categories, accs)))
        print(f"      spread {max(accs) - min(accs):.0f} points")
    best = max(("conf_gap12", "conf_gap110", "conf_top"),
               key=lambda s: abs(np.corrcoef(df[s].fillna(0),
                                             df.rank1_correct)[0, 1]))
    r = np.corrcoef(df[best].fillna(0), df.rank1_correct)[0, 1]
    print(f"\n    strongest signal: {best} (r = {r:+.3f})")
    print(f"    A usable confidence signal lets an operator know which")
    print(f"    results to trust, which matters more than raw accuracy when")
    print(f"    time is short.")

    # ------------------------------------------------------------- 5
    print("\n" + "=" * 76)
    print("5.  DOES THE FUSION METHOD MATTER?")
    print("=" * 76)
    for lab, c_ in (("weighted z-scores", "ap_fused"),
                    ("reciprocal rank fusion", "ap_rrf")):
        m, lo, hi = ci(df[c_].values * 100)
        print(f"    {lab:<26}{m:6.2f}  [{lo:.2f}, {hi:.2f}]")
    m3, lo3, hi3 = ci((df.ap_fused - df.ap_rrf).values * 100)
    print(f"    difference                {m3:+6.2f}  [{lo3:+.2f}, {hi3:+.2f}]")
    print(f"\n    Rank fusion uses only the ordering, so it is immune to")
    print(f"    score-scale problems but discards the margin between")
    print(f"    candidates. Which wins is an empirical question.")

    out = cache / "phase6_deep.csv"
    df.drop(columns=["cap"]).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
