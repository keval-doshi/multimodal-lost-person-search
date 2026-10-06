"""
Phase 3d - Retrieval baselines at native resolution.

Produces the first two research numbers: how well a photograph alone
retrieves the right person, and how well a written description alone does.
These are the top row of the main result table; Phase 4 repeats them at every
resolution level and Phase 5 adds the fused methods.

  python phase3_baseline.py --root .\\dataset

Protocol
--------
Gallery      detections from the official PRW test frames, including the
             ones that matched no annotation. Those are distractors a real
             search has to rank past, so removing them would flatter the
             result.
Queries      the 2,057 hand-drawn query boxes (450 identities).
Captions      4,114 descriptions, each scored as its own query, which is the
             convention in text-based person search.
Junk         gallery crops from the query's own camera are removed before
             ranking: matching a person to another shot from the same camera
             is trivial and inflates the score.
Metrics      Rank-1, Rank-5, Rank-10, mean average precision.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_split(prw: Path, name: str):
    from scipy.io import loadmat
    p = prw / name
    if not p.exists():
        return None
    m = loadmat(str(p))
    arr = next((v for k, v in m.items()
                if not k.startswith("__") and isinstance(v, np.ndarray) and v.size),
               None)
    if arr is None:
        return None
    out = set()
    for v in np.ravel(arr):
        s = str(v[0]) if isinstance(v, np.ndarray) and v.size else str(v)
        s = s.strip()
        out.add(s if s.endswith(".jpg") else s + ".jpg")
    return out


def evaluate(qf, q_pid, q_cam, gf, g_pid, g_cam, topk=(1, 5, 10), chunk=512):
    """Cosine ranking with same-camera junk removal. Features are unit norm."""
    max_k = max(topk)
    cmcs, aps = [], []

    for s in range(0, len(qf), chunk):
        e = min(s + chunk, len(qf))
        sim = qf[s:e] @ gf.T
        order = np.argsort(-sim, axis=1)

        for r in range(e - s):
            i = s + r
            o = order[r]
            keep = ~(g_cam[o] == q_cam[i])          # drop same-camera
            o = o[keep]
            match = (g_pid[o] == q_pid[i]).astype(np.int32)
            if match.sum() == 0:
                continue

            c = match.cumsum()
            c[c > 1] = 1
            if len(c) < max_k:
                c = np.concatenate([c, np.full(max_k - len(c), c[-1])])
            cmcs.append(c[:max_k])

            pos = np.where(match == 1)[0]
            aps.append(float(np.mean([(j + 1) / (p + 1)
                                      for j, p in enumerate(pos)])))

    if not cmcs:
        return {f"Rank-{k}": 0.0 for k in topk} | {"mAP": 0.0, "queries": 0}
    cmc = np.stack(cmcs).mean(0) * 100
    return {f"Rank-{k}": float(cmc[k - 1]) for k in topk} | \
           {"mAP": float(np.mean(aps) * 100), "queries": len(aps)}


def show(name, r):
    print(f"\n{name}")
    print(f"  Rank-1  {r['Rank-1']:6.2f} %")
    print(f"  Rank-5  {r['Rank-5']:6.2f} %")
    print(f"  Rank-10 {r['Rank-10']:6.2f} %")
    print(f"  mAP     {r['mAP']:6.2f} %")
    print(f"  queries scored: {r['queries']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--gallery", default="test", choices=["test", "all"],
                    help="'test' follows the official PRW split")
    ap.add_argument("--min-height", type=int, default=0,
                    help="drop gallery crops shorter than this many pixels")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    prw = root / "PRW"
    cache = root.parent / "cache"

    for f in ("gallery.npz", "query.npz", "captions.npz"):
        if not (cache / f).exists():
            sys.exit(f"ERROR: {cache / f} missing. Run phase3_encode.py first.")

    G = np.load(cache / "gallery.npz", allow_pickle=True)
    Q = np.load(cache / "query.npz", allow_pickle=True)
    C = np.load(cache / "captions.npz", allow_pickle=True)

    g_frames = G["frames"]
    mask = np.ones(len(g_frames), bool)

    if a.gallery == "test":
        sp = load_split(prw, "frame_test.mat")
        if sp:
            mask &= np.array([f in sp for f in g_frames])
            print(f"Gallery restricted to test frames: {mask.sum()} of {len(mask)} crops")
        else:
            print("WARNING: frame_test.mat unreadable, using all frames")
    if a.min_height:
        mask &= (G["height"] >= a.min_height)
        print(f"After height >= {a.min_height}px: {mask.sum()} crops")

    g_os, g_ir = G["osnet"][mask], G["irra"][mask]
    g_pid, g_cam = G["ids"][mask], G["cams"][mask]

    known = int((g_pid > 0).sum())
    print(f"\nGallery: {len(g_pid)} crops, {known} with an identity "
          f"({known / len(g_pid) * 100:.1f}%), "
          f"{len(g_pid) - known} distractors")
    print(f"         {len(set(g_pid[g_pid > 0].tolist()))} distinct identities")

    q_os, q_ir = Q["osnet"], Q["irra"]
    q_pid, q_cam = Q["ids"], Q["cams"]
    print(f"Queries: {len(q_pid)} boxes, {len(set(q_pid.tolist()))} identities")

    # captions carry no camera of their own; inherit it from the paired
    # query box so junk removal is applied identically to both branches
    frame_to_cam = {}
    for f, c in zip(Q["frames"], q_cam):
        frame_to_cam[str(f)] = c
    c_frames = [str(f) for f in C["frames"]]
    c_cam = np.array([frame_to_cam.get(f, -1) for f in c_frames], np.int32)
    c_pid = C["ids"]
    unmatched = int((c_cam == -1).sum())
    if unmatched:
        print(f"  note: {unmatched} captions had no paired query box; "
              f"camera inferred from the frame name")
        for i, f in enumerate(c_frames):
            if c_cam[i] == -1:
                c_cam[i] = int(f.split("s")[0][1:])
    print(f"Captions: {len(c_pid)}, {len(set(c_pid.tolist()))} identities")

    print("\n" + "=" * 58)
    print("NATIVE-RESOLUTION BASELINES")
    print("=" * 58)

    r_img = evaluate(q_os, q_pid, q_cam, g_os, g_pid, g_cam)
    show("PHOTOGRAPH ONLY   (OSNet appearance)", r_img)

    r_txt = evaluate(C["irra"], c_pid, c_cam, g_ir, g_pid, g_cam)
    show("DESCRIPTION ONLY  (IRRA text vs image)", r_txt)

    r_qim = evaluate(q_ir, q_pid, q_cam, g_ir, g_pid, g_cam)
    show("PHOTOGRAPH ONLY   (IRRA image branch, for reference)", r_qim)

    # the join Phase 5 depends on: every caption needs a paired query box
    qkeys = {(int(p), str(f)) for p, f in zip(q_pid, Q["frames"])}
    ckeys = {(int(p), str(f)) for p, f in zip(c_pid, C["frames"])}
    both = qkeys & ckeys
    print("\n" + "-" * 58)
    print(f"Query pairing for Phase 5: {len(both)} of {len(ckeys)} "
          f"caption entries have a matching query box")
    if len(both) < len(ckeys) * 0.95:
        print("  WARNING: many captions cannot be paired with an image query.")
        print("  Fusion needs both, so check the id/frame join before Phase 5.")
    else:
        print("  The image and text branches can be fused per query.")

    out = cache / "phase3_baselines.csv"
    import csv
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "Rank-1", "Rank-5", "Rank-10", "mAP", "queries"])
        for name, r in (("photo_osnet", r_img), ("text_irra", r_txt),
                        ("photo_irra", r_qim)):
            w.writerow([name] + [f"{r[k]:.2f}" for k in
                                 ("Rank-1", "Rank-5", "Rank-10", "mAP")]
                       + [r["queries"]])
    print(f"\nSaved {out}")

    print("\n" + "=" * 58)
    print("READING THE RESULT")
    print("=" * 58)
    d = r_img["Rank-1"] - r_txt["Rank-1"]
    print(f"At native resolution the photograph leads the description by "
          f"{d:+.1f} points at Rank-1.")
    print("That gap is the starting point of the study. Phase 4 repeats this")
    print("at each resolution level; the research question is how much of the")
    print("gap survives as the gallery crops shrink.")


if __name__ == "__main__":
    main()
