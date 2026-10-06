"""
Phase 6c - Deep analysis.

Five questions the headline numbers leave unanswered. Each is answered from
the cached features, so this runs in a few minutes with no GPU.

  python phase6_analysis.py --root .\\dataset

1  COMPLEMENTARITY
   Phase 5 showed fusion beats both sources despite them being equally
   strong, and inferred complementarity. This measures it directly: do the
   photograph and the description retrieve the SAME correct people, or
   different ones? Overlapping sets would mean the gain comes from noise
   reduction instead, which is a different mechanism.

2  WHERE THE GAIN LIVES
   Phase 4 found the description gains value as crops shrink. If fusion also
   helps more for small query boxes, the two findings are one argument.
   Stratified by query size, by how many times the person appears, and by
   how well the photograph alone did.

3  CAMERA BIAS
   PRW's cameras differ greatly in size: c6 contributed 2,898 frames against
   1,206 for c4. If recovered sightings track gallery size rather than real
   appearances, the timeline is systematically misleading.

4  PHRASING SENSITIVITY
   Every test identity carries two independently written descriptions. Same
   person, same gallery, different wording. The spread between them measures
   how much the result depends on how the witness phrased it.

5  FAILURE MODES
   A wrong result is either an unidentified detection or the wrong person.
   These are different problems with different remedies.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

ALPHA = 0.45
TOPK = 10


def zscore_rows(S):
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True) + 1e-8
    return (S - mu) / sd


def ap_of(scores, rel, valid):
    s, r = scores[valid], rel[valid]
    n = int(r.sum())
    if n == 0:
        return np.nan
    o = np.argsort(-s, kind="stable")
    ro = r[o]
    pos = np.flatnonzero(ro)
    return float(np.mean((np.arange(n) + 1) / (pos + 1)))


def topk_ids(scores, valid, k):
    """Indices of the top k valid gallery entries, in rank order."""
    idx = np.flatnonzero(valid)
    o = idx[np.argsort(-scores[idx], kind="stable")[:k]]
    return o


def boot_ci(v, reps=2000, seed=0):
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(reps, len(v)))].mean(axis=1)
    return v.mean(), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    P = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_argument("--root", default="dataset")
    P.add_argument("--alpha", type=float, default=ALPHA)
    P.add_argument("--topk", type=int, default=TOPK)
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
    g_h = G["height"][mask]

    q_os, q_ir = Q["osnet"].astype(np.float32), Q["irra"].astype(np.float32)
    q_pid, q_cam, q_frames = Q["ids"], Q["cams"], Q["frames"]

    # captions grouped by their paired query box
    key = {(int(p), str(f)): i for i, (p, f) in enumerate(zip(q_pid, q_frames))}
    caps_of = {}
    for i, (p, f) in enumerate(zip(C["ids"], C["frames"])):
        r = key.get((int(p), str(f)))
        if r is not None:
            caps_of.setdefault(r, []).append(i)
    c_ir = C["irra"].astype(np.float32)
    c_text = C["text"]

    print(f"Gallery {len(g_pid)} crops | queries {len(q_pid)} | "
          f"captions {len(c_ir)}")
    print(f"alpha {a_.alpha}, top-{a_.topk} for set comparisons\n")

    Z_IMG = zscore_rows(q_os @ g_os.T)
    Z_TXT_ALL = zscore_rows(c_ir @ g_ir.T)

    rows = []
    for qi in range(len(q_pid)):
        ci = caps_of.get(qi)
        if not ci:
            continue
        rel = (g_pid == q_pid[qi])
        valid = (g_cam != q_cam[qi])
        n_true = int((rel & valid).sum())
        if n_true == 0:
            continue

        zi = Z_IMG[qi]
        zt = Z_TXT_ALL[ci[0]]
        zf = a_.alpha * zi + (1 - a_.alpha) * zt

        ap_i, ap_t, ap_f = (ap_of(zi, rel, valid), ap_of(zt, rel, valid),
                            ap_of(zf, rel, valid))

        ti = topk_ids(zi, valid, a_.topk)
        tt = topk_ids(zt, valid, a_.topk)
        tf = topk_ids(zf, valid, a_.topk)
        ci_set = set(ti[rel[ti]].tolist())
        ct_set = set(tt[rel[tt]].tolist())
        cf_set = set(tf[rel[tf]].tolist())

        # failure taxonomy on the fused list
        wrong = tf[~rel[tf]]
        n_distract = int((g_pid[wrong] < 0).sum())
        n_wrongid = int(len(wrong) - n_distract)

        # phrasing spread: the two independently written captions
        ap_c = [ap_of(a_.alpha * zi + (1 - a_.alpha) * Z_TXT_ALL[c], rel, valid)
                for c in ci[:2]]
        ap_txt_only = [ap_of(Z_TXT_ALL[c], rel, valid) for c in ci[:2]]

        rows.append({
            "qi": qi, "pid": int(q_pid[qi]), "cam": int(q_cam[qi]),
            "n_true": n_true,
            "n_cams_true": int(len(set(g_cam[rel & valid].tolist()))),
            "q_height": int(np.median(g_h[rel]) if rel.any() else 0),
            "ap_img": ap_i, "ap_txt": ap_t, "ap_fused": ap_f,
            "gain": ap_f - ap_i,
            "hit_img": len(ci_set), "hit_txt": len(ct_set),
            "hit_fused": len(cf_set),
            "overlap": len(ci_set & ct_set),
            "union": len(ci_set | ct_set),
            "txt_only_finds": len(ct_set - ci_set),
            "img_only_finds": len(ci_set - ct_set),
            "distractors_in_top": n_distract,
            "wrong_id_in_top": n_wrongid,
            "cap_len": int(np.mean([len(str(c_text[c]).split()) for c in ci])),
            "ap_cap1": ap_c[0], "ap_cap2": ap_c[1] if len(ap_c) > 1 else np.nan,
            "ap_txt1": ap_txt_only[0],
            "ap_txt2": ap_txt_only[1] if len(ap_txt_only) > 1 else np.nan,
        })

    df = pd.DataFrame(rows)
    print(f"Analysed {len(df)} queries\n")

    # ---------------------------------------------------------------- 1
    print("=" * 74)
    print("1.  IS THE GAIN COMPLEMENTARITY, OR NOISE REDUCTION?")
    print("=" * 74)
    d = df[(df.hit_img > 0) | (df.hit_txt > 0)]
    print(f"  Among {len(d)} queries where at least one source found something")
    print(f"  in its top {a_.topk}:\n")
    print(f"    correct found by the photograph      {d.hit_img.mean():.2f}")
    print(f"    correct found by the description     {d.hit_txt.mean():.2f}")
    print(f"    found by both (overlap)              {d.overlap.mean():.2f}")
    print(f"    union of the two                     {d.union.mean():.2f}")
    print(f"    found ONLY by the description        {d.txt_only_finds.mean():.2f}")
    print(f"    found ONLY by the photograph         {d.img_only_finds.mean():.2f}")
    jac = (d.overlap / d.union.replace(0, np.nan)).mean()
    print(f"\n    mean overlap as a fraction of union: {jac:.3f}")
    uniq = (d.txt_only_finds.sum() /
            max(d.hit_txt.sum(), 1))
    print(f"    of everything the description found, "
          f"{uniq * 100:.0f}% was missed by the photograph")
    if jac < 0.5:
        print("\n    -> the two sources retrieve largely DIFFERENT correct")
        print("       people. The gain is genuine complementarity, not")
        print("       averaging away noise.")
    else:
        print("\n    -> the two sources retrieve largely the SAME people, so")
        print("       the gain comes from reinforcing shared evidence rather")
        print("       than from complementary coverage.")

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 74)
    print("2.  WHERE DOES THE FUSION GAIN LIVE?")
    print("=" * 74)

    def strat(col, bins, labels, title):
        print(f"\n  by {title}")
        print(f"    {'band':<16}{'n':>7}{'img AP':>10}{'fused AP':>11}{'gain':>9}")
        print("    " + "-" * 54)
        b = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
        for lab in labels:
            s = df[b == lab]
            if len(s) < 20:
                continue
            print(f"    {str(lab):<16}{len(s):>7}{s.ap_img.mean() * 100:>9.1f}%"
                  f"{s.ap_fused.mean() * 100:>10.1f}%{s.gain.mean() * 100:>8.1f}")

    strat("q_height", [0, 100, 150, 220, 10 ** 6],
          ["under 100px", "100-150px", "150-220px", "over 220px"],
          "how large the person appears")
    strat("n_true", [0, 10, 25, 60, 10 ** 6],
          ["under 10", "10-25", "25-60", "over 60"],
          "how many times the person appears in the gallery")
    strat("ap_img", [-.01, .1, .3, .6, 1.01],
          ["under 10%", "10-30%", "30-60%", "over 60%"],
          "how well the photograph alone did")

    hard = df[df.ap_img < 0.3]
    easy = df[df.ap_img >= 0.6]
    print(f"\n  Fusion adds {hard.gain.mean() * 100:+.1f} points where the "
          f"photograph struggled (n={len(hard)})")
    print(f"  and {easy.gain.mean() * 100:+.1f} points where it already "
          f"worked well (n={len(easy)})")
    print(f"  Queries where fusion HURT: {(df.gain < -0.01).mean() * 100:.1f}%")

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 74)
    print("3.  IS THE TIMELINE BIASED BY CAMERA SIZE?")
    print("=" * 74)
    print(f"    {'camera':<9}{'gallery':>9}{'share':>8}{'true appear':>13}"
          f"{'in top-10':>11}{'recall':>9}")
    print("    " + "-" * 60)
    tot = len(g_pid)
    for c in sorted(set(g_cam.tolist())):
        n_gal = int((g_cam == c).sum())
        appear = found = 0
        for r in rows:
            rel = (g_pid == r["pid"]) & (g_cam != r["cam"]) & (g_cam == c)
            if rel.any():
                appear += 1
                zf = (a_.alpha * Z_IMG[r["qi"]] +
                      (1 - a_.alpha) * Z_TXT_ALL[caps_of[r["qi"]][0]])
                tf = topk_ids(zf, g_cam != r["cam"], a_.topk)
                if rel[tf].any():
                    found += 1
        rec = found / appear * 100 if appear else 0
        print(f"    c{c:<8}{n_gal:>9}{n_gal / tot * 100:>7.1f}%{appear:>13}"
              f"{found:>11}{rec:>8.1f}%")
    print("\n    If recall tracked gallery size rather than real sightings,")
    print("    the largest cameras would show the highest recall. Compare the")
    print("    share column with the recall column.")

    # ---------------------------------------------------------------- 4
    print("\n" + "=" * 74)
    print("4.  HOW MUCH DOES THE WORDING MATTER?")
    print("=" * 74)
    dd = df.dropna(subset=["ap_txt1", "ap_txt2"])
    diff_t = (dd.ap_txt1 - dd.ap_txt2).abs() * 100
    diff_f = (dd.ap_cap1 - dd.ap_cap2).abs() * 100
    print(f"  {len(dd)} people carry two independently written descriptions.")
    print(f"  Same person, same gallery, different wording.\n")
    print(f"    description alone, mean absolute difference in AP "
          f"{diff_t.mean():6.2f} points")
    print(f"    median {np.median(diff_t):.2f}, "
          f"90th percentile {np.percentile(diff_t, 90):.2f}, "
          f"max {diff_t.max():.2f}")
    print(f"\n    after fusing with the photograph        "
          f"{diff_f.mean():6.2f} points")
    print(f"    median {np.median(diff_f):.2f}, "
          f"90th percentile {np.percentile(diff_f, 90):.2f}")
    red = (1 - diff_f.mean() / max(diff_t.mean(), 1e-9)) * 100
    print(f"\n    fusion reduces sensitivity to wording by {red:.0f}%")
    print("    -> a practical argument for collecting both: the photograph")
    print("       stabilises a result that would otherwise depend heavily on")
    print("       how the witness happened to phrase the description.")

    cl = df.dropna(subset=["ap_txt"])
    b = pd.cut(cl.cap_len, [0, 18, 24, 30, 10 ** 6],
               labels=["under 18 words", "18-24", "24-30", "over 30"])
    print(f"\n  by description length")
    print(f"    {'band':<18}{'n':>7}{'text AP':>10}")
    print("    " + "-" * 36)
    for lab in b.cat.categories:
        s = cl[b == lab]
        if len(s) >= 20:
            print(f"    {str(lab):<18}{len(s):>7}{s.ap_txt.mean() * 100:>9.1f}%")

    # ---------------------------------------------------------------- 5
    print("\n" + "=" * 74)
    print("5.  WHAT ARE THE FAILURES MADE OF?")
    print("=" * 74)
    tot_wrong = df.distractors_in_top.sum() + df.wrong_id_in_top.sum()
    print(f"  Across {len(df)} queries, top-{a_.topk} contained "
          f"{int(tot_wrong)} incorrect results:")
    print(f"    unidentified detections (no ground-truth identity)  "
          f"{int(df.distractors_in_top.sum()):>7}  "
          f"({df.distractors_in_top.sum() / tot_wrong * 100:.1f}%)")
    print(f"    genuinely the wrong person                          "
          f"{int(df.wrong_id_in_top.sum()):>7}  "
          f"({df.wrong_id_in_top.sum() / tot_wrong * 100:.1f}%)")
    frac_d = (g_pid < 0).mean()
    print(f"\n  Unidentified crops are {frac_d * 100:.1f}% of the gallery but "
          f"{df.distractors_in_top.sum() / tot_wrong * 100:.1f}% of the errors.")
    if df.distractors_in_top.sum() / tot_wrong > frac_d * 1.2:
        print("  They are OVER-represented among errors: unlabelled people and")
        print("  false detections attract matches more than chance would give.")
    else:
        print("  They are not over-represented, so the dominant failure is")
        print("  confusing one identified person with another.")

    m, lo, hi = boot_ci(df.gain.values * 100)
    print(f"\n  Mean fusion gain in AP: {m:+.2f} [{lo:+.2f}, {hi:+.2f}]")

    out = cache / "phase6_analysis.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
