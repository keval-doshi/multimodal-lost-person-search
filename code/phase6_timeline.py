"""
Phase 6a - Multi-camera sighting timeline.

Applies the tuned fusion from Phase 5 across all six cameras and turns the
ranked results into what an operator actually needs: which cameras saw the
person, in what order, and with what confidence.

  # evaluate the whole test set
  python phase6_timeline.py --root .\\dataset --evaluate

  # inspect one query
  python phase6_timeline.py --root .\\dataset --query 479

Configuration
-------------
Appearance branch with alpha = 0.45, the tuned constant from Phase 5. That
configuration gained +10.43 mAP over the photograph alone (95% CI [+9.79,
+11.09]). The quality-aware rule is not used: Phase 5 established it adds
nothing over the tuned constant.

On timing
---------
PRW frame names encode camera, segment and frame index, for example
c1s3_016471. The dataset documentation states segments were generated
automatically during collection, and the six cameras share no clock. Frame
indices therefore order sightings WITHIN a segment but cannot be compared
across segments or cameras.

The output is consequently an ordered sequence of sightings per camera, not
a set of synchronised times. This is a property of the dataset, not of the
method: with time-stamped footage the same pipeline would produce clock
times directly.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

ALPHA = 0.45          # tuned constant, appearance branch, Phase 5
BRANCH = "osnet"


def zscore_rows(S):
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True) + 1e-8
    return (S - mu) / sd


def parse_frame(name):
    """c1s3_016471.jpg -> (camera, segment, frame index)"""
    stem = str(name).replace(".jpg", "")
    cam_seg, _, idx = stem.partition("_")
    cam, _, seg = cam_seg.partition("s")
    return int(cam[1:]), int(seg), int(idx)


def build_timeline(scores, g_frames, g_cams, g_pid, q_cam,
                   top_k=50, min_gap=25):
    """
    Turn a ranked gallery into a sighting list.

    Consecutive frames of the same person in the same segment are one
    sighting, not many, so detections within `min_gap` frames of an existing
    sighting in the same segment are folded into it and only the strongest
    is kept.
    """
    order = np.argsort(-scores)
    order = order[g_cams[order] != q_cam]          # exclude the query's camera
    sightings = []
    for gi in order[:top_k * 4]:
        cam, seg, idx = parse_frame(g_frames[gi])
        merged = False
        for s in sightings:
            if s["camera"] == cam and s["segment"] == seg and \
                    abs(s["frame"] - idx) <= min_gap:
                if scores[gi] > s["score"]:
                    s.update(frame=idx, score=float(scores[gi]),
                             gallery_index=int(gi), true_id=int(g_pid[gi]))
                s["detections"] += 1
                merged = True
                break
        if not merged:
            sightings.append({"camera": cam, "segment": seg, "frame": idx,
                              "score": float(scores[gi]),
                              "gallery_index": int(gi),
                              "true_id": int(g_pid[gi]), "detections": 1})
        if len(sightings) >= top_k:
            break
    return sightings


def show_timeline(sightings, q_id, n=12):
    print(f"\n{'rank':<6}{'camera':<9}{'segment':<10}{'frame':<10}"
          f"{'score':<9}{'correct':<9}")
    print("-" * 58)
    for i, s in enumerate(sightings[:n], 1):
        ok = "yes" if s["true_id"] == q_id else \
             ("distractor" if s["true_id"] < 0 else "no")
        print(f"{i:<6}{'c' + str(s['camera']):<9}{s['segment']:<10}"
              f"{s['frame']:<10}{s['score']:<9.3f}{ok:<9}")

    by_cam = {}
    for s in sightings[:n]:
        by_cam.setdefault(s["camera"], []).append(s)
    print(f"\nCameras with a candidate sighting: "
          f"{', '.join('c' + str(c) for c in sorted(by_cam))}")
    for c in sorted(by_cam):
        segs = sorted({s["segment"] for s in by_cam[c]})
        best = max(s["score"] for s in by_cam[c])
        print(f"  c{c}: {len(by_cam[c])} sighting(s) across segment(s) "
              f"{', '.join(map(str, segs))}, best score {best:.3f}")


def main():
    P = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_argument("--root", default="dataset")
    P.add_argument("--query", type=int, default=0,
                   help="identity to inspect; 0 picks one automatically")
    P.add_argument("--evaluate", action="store_true",
                   help="score timeline quality over the whole test set")
    P.add_argument("--alpha", type=float, default=ALPHA)
    P.add_argument("--top-k", type=int, default=20)
    a_ = P.parse_args()

    cache = Path(a_.root).resolve().parent / "cache"
    G = np.load(cache / "gallery.npz", allow_pickle=True)
    Q = np.load(cache / "query.npz", allow_pickle=True)
    C = np.load(cache / "captions.npz", allow_pickle=True)

    from phase3_baseline import load_split
    sp = load_split(Path(a_.root).resolve() / "PRW", "frame_test.mat")
    mask = np.array([f in sp for f in G["frames"]]) if sp else np.ones(len(G["frames"]), bool)

    g_os = G["osnet"][mask].astype(np.float32)
    g_ir = G["irra"][mask].astype(np.float32)
    g_pid, g_cam = G["ids"][mask], G["cams"][mask]
    g_frames = G["frames"][mask]
    print(f"Gallery: {len(g_pid)} crops across "
          f"{len(set(g_cam.tolist()))} cameras")

    key = {(int(p), str(f)): i for i, (p, f) in enumerate(zip(Q["ids"], Q["frames"]))}
    cap_of = {}
    for i, (p, f) in enumerate(zip(C["ids"], C["frames"])):
        if (int(p), str(f)) in key:
            cap_of.setdefault(key[(int(p), str(f))], []).append(i)

    q_os, q_ir = Q["osnet"].astype(np.float32), Q["irra"].astype(np.float32)
    q_pid, q_cam, q_frames = Q["ids"], Q["cams"], Q["frames"]
    c_ir = C["irra"].astype(np.float32)

    # ---------------------------------------------------------- evaluate
    if a_.evaluate:
        print(f"\nEvaluating timelines over {len(q_pid)} queries "
              f"(alpha {a_.alpha})\n")
        rows = []
        for qi in range(len(q_pid)):
            ci = cap_of.get(qi)
            if not ci:
                continue
            z_img = zscore_rows((q_os[qi:qi + 1] @ g_os.T))[0]
            z_txt = zscore_rows((c_ir[ci[0]:ci[0] + 1] @ g_ir.T))[0]
            fused = a_.alpha * z_img + (1 - a_.alpha) * z_txt

            st = build_timeline(fused, g_frames, g_cam, g_pid,
                                q_cam[qi], top_k=a_.top_k)
            if not st:
                continue
            correct = [s for s in st if s["true_id"] == q_pid[qi]]
            true_cams = set(g_cam[(g_pid == q_pid[qi]) &
                                  (g_cam != q_cam[qi])].tolist())
            found_cams = {s["camera"] for s in correct}
            rows.append({
                "query_id": int(q_pid[qi]),
                "sightings": len(st),
                "correct": len(correct),
                "precision": len(correct) / len(st),
                "first_correct_rank": next(
                    (i for i, s in enumerate(st, 1)
                     if s["true_id"] == q_pid[qi]), None),
                "cameras_true": len(true_cams),
                "cameras_found": len(found_cams & true_cams),
            })

        df = pd.DataFrame(rows)
        out = cache / "phase6_timeline_eval.csv"
        df.to_csv(out, index=False)

        print("=" * 62)
        print(f"TIMELINE QUALITY  (top {a_.top_k} sightings per query)")
        print("=" * 62)
        print(f"  queries evaluated              {len(df)}")
        print(f"  mean precision of the list     "
              f"{df['precision'].mean() * 100:.1f}%")
        hit = df["first_correct_rank"].notna()
        print(f"  at least one correct sighting  {hit.mean() * 100:.1f}%")
        print(f"  median rank of the first hit   "
              f"{df.loc[hit, 'first_correct_rank'].median():.0f}")
        cc = df[df["cameras_true"] > 0]
        print(f"\n  cross-camera coverage")
        print(f"    mean cameras the person truly appears on   "
              f"{cc['cameras_true'].mean():.2f}")
        print(f"    mean cameras recovered                     "
              f"{cc['cameras_found'].mean():.2f}")
        print(f"    coverage                                   "
              f"{(cc['cameras_found'].sum() / cc['cameras_true'].sum()) * 100:.1f}%")
        print(f"\n  A perfect system would recover every camera the person")
        print(f"  appeared on. Coverage is the fraction it actually finds,")
        print(f"  and it is the number that matters for building a route.")
        print(f"\nSaved {out}")
        return

    # ---------------------------------------------------------- single query
    qi = None
    if a_.query:
        cands = [i for i in range(len(q_pid))
                 if q_pid[i] == a_.query and i in cap_of]
        if not cands:
            sys.exit(f"No query box with captions for identity {a_.query}")
        qi = cands[0]
    else:
        for i in range(len(q_pid)):
            if i in cap_of and (g_pid == q_pid[i]).sum() >= 6:
                qi = i
                break
    if qi is None:
        sys.exit("No suitable query found")

    ci = cap_of[qi][0]
    print("\n" + "=" * 62)
    print(f"QUERY  identity {q_pid[qi]}   from {q_frames[qi]} (camera c{q_cam[qi]})")
    print("=" * 62)
    print(f"description: {C['text'][ci]}")

    z_img = zscore_rows(q_os[qi:qi + 1] @ g_os.T)[0]
    z_txt = zscore_rows(c_ir[ci:ci + 1] @ g_ir.T)[0]
    fused = a_.alpha * z_img + (1 - a_.alpha) * z_txt

    for lab, sc in (("photograph only", z_img), ("description only", z_txt),
                    (f"fused (alpha {a_.alpha})", fused)):
        st = build_timeline(sc, g_frames, g_cam, g_pid, q_cam[qi],
                            top_k=a_.top_k)
        corr = sum(1 for s in st if s["true_id"] == q_pid[qi])
        print(f"\n--- {lab}: {corr}/{len(st)} correct in the top {len(st)}")
        if lab.startswith("fused"):
            show_timeline(st, int(q_pid[qi]))

    true_cams = sorted(set(g_cam[(g_pid == q_pid[qi]) &
                                 (g_cam != q_cam[qi])].tolist()))
    print(f"\nGround truth: this person also appears on camera(s) "
          f"{', '.join('c' + str(c) for c in true_cams)}")


if __name__ == "__main__":
    main()
