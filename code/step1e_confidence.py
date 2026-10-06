"""
Step 1e - Confirmatory test of the confidence signal.

The exploratory analysis in Phase 6 identified the TOP fused score as the
strongest confidence signal (Rank-1 rose from 27% to 94% across quintiles),
but that analysis used all queries, including tuning identities. Step 1d
tested a DIFFERENT signal by mistake (the gap between the top two scores).

This script tests the pre-specified signal properly:
  * signal: top fused score, fusion as in Phase 6 (single description, OSNet,
    alpha tuned on validation, no re-ranking) - the configuration in which
    it was found
  * quintile thresholds fixed on VALIDATION identities
  * Rank-1 per band measured on HELD-OUT identities, identity-level intervals
  * also reported, for the complete system (two photos, two descriptions,
    re-ranking), so the thesis can say whether the signal transfers

It is the only re-test run; no further signals are tried.

  python step1e_confidence.py --root .\\dataset     # ~5 min
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from step1_corrected_eval import zrow, unit, rerank, metrics, cboot


def main():
    P = argparse.ArgumentParser(); P.add_argument("--root", default="dataset")
    P.add_argument("--seed", type=int, default=0); P.add_argument("--limit", type=int, default=0)
    a = P.parse_args()
    from phase3_baseline import load_split
    from phase3_encode import load_gt_boxes
    root = Path(a.root).resolve(); prw = root / "PRW"
    cache = root.parent / "cache"; out = cache / "step1"
    curves = pd.read_csv(out / "alpha_curves.csv")
    s = curves[(curves.gallery == "native") & (curves.branch == "osnet")]
    alpha = float(s[s.ap >= s.ap.max() - 1e-9].alpha.max())

    G = np.load(cache / "gallery.npz", allow_pickle=True)
    Q = np.load(cache / "query.npz", allow_pickle=True)
    C = np.load(cache / "captions.npz", allow_pickle=True)
    tf = load_split(prw, "frame_test.mat")
    mask = np.array([f in tf for f in G["frames"]])
    GP, GC = G["ids"][mask], G["cams"][mask]
    Go, Gi = G["osnet"][mask].astype(np.float32), G["irra"][mask].astype(np.float32)
    gt = load_gt_boxes(prw, sorted(tf)); gtc = {}
    for f, (_, ids) in gt.items():
        cam = int(f.split("s")[0][1:])
        for p in ids:
            if p > 0: gtc[(int(p), cam)] = gtc.get((int(p), cam), 0) + 1

    q_os = Q["osnet"].astype(np.float32); q_pid, q_cam, q_fr = Q["ids"], Q["cams"], Q["frames"]
    c_ir = C["irra"].astype(np.float32)
    key = {(int(p), str(f)): i for i, (p, f) in enumerate(zip(q_pid, q_fr))}
    caps = {}
    for i, (p, f) in enumerate(zip(C["ids"], C["frames"])):
        r = key.get((int(p), str(f)))
        if r is not None: caps.setdefault(r, []).append(i)
    by_id = {}
    for i in range(len(q_pid)):
        if i in caps: by_id.setdefault(int(q_pid[i]), []).append(i)
    qs = [i for i in range(len(q_pid)) if i in caps]
    ids = np.unique(q_pid[qs]); sh = ids.copy(); np.random.default_rng(a.seed).shuffle(sh)
    val_ids = set(sh[:len(sh) // 2].tolist())

    rows = []
    for qi in qs:
        pid = int(q_pid[qi])
        grp = [qi] + [j for j in by_id[pid] if j != qi][:1]
        excl = sorted(set(int(c) for c in q_cam[grp]))
        n_gt = sum(v for (p, c), v in gtc.items() if p == pid and c not in excl)
        if n_gt == 0: continue
        idx = np.flatnonzero(~((GP == pid) & np.isin(GC, excl))); rel = GP[idx] == pid
        cl = caps[qi]
        # configuration in which the signal was found: one photo, one description
        f1 = alpha * zrow(q_os[qi] @ Go[idx].T) + (1 - alpha) * zrow(c_ir[cl[0]] @ Gi[idx].T)
        # complete system
        f2 = alpha * zrow(unit(q_os[grp].mean(0)) @ Go[idx].T) + \
             (1 - alpha) * zrow(unit(c_ir[cl[:2]].mean(0)) @ Gi[idx].T)
        sys_scores = rerank(f2, Go[idx])
        rows.append({"qi": qi, "pid": pid, "split": "val" if pid in val_ids else "test",
                     "top_single": float(f1.max()), "r1_single": metrics(f1, rel, n_gt)[3],
                     "top_system": float(f2.max()), "r1_system": metrics(sys_scores, rel, n_gt)[3]})
    df = pd.DataFrame(rows)
    if a.limit:
        df = pd.concat([df[df.split == "val"].head(a.limit), df[df.split == "test"].head(a.limit)])
    df.to_csv(out / "e_confidence_perquery.csv", index=False)

    V, T = df[df.split == "val"], df[df.split == "test"]
    lines = [f"alpha {alpha} | validation {len(V)} | held-out {len(T)} queries",
             "Rank-1 on HELD-OUT identities per confidence band (thresholds from validation)"]
    for sig, r1, lab in (("top_single", "r1_single", "fused retrieval, one photo + one description"),
                         ("top_system", "r1_system", "complete system")):
        edges = np.quantile(V[sig], [0.2, 0.4, 0.6, 0.8])
        band = np.digitize(T[sig], edges)
        parts = []
        for k, nm in enumerate(["lowest", "second", "middle", "fourth", "highest"]):
            s_ = T[band == k]
            m = cboot(s_[r1].values * 100, s_.pid.values)
            parts.append(f"{nm} {m[0]:.0f}% [{m[1]:.0f},{m[2]:.0f}] n={len(s_)}")
        lines.append(f"\n{lab}:\n  " + "\n  ".join(parts))
        lines.append(f"  overall Rank-1 {T[r1].mean() * 100:.1f}%")
    txt = "\n".join(lines); print(txt)
    (out / "e_confidence.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
