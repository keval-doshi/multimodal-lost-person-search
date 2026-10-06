"""
Diagnostic for Phase 8. Prints exactly what happens to the first few queries
so an empty result can be traced to its cause.

  python phase8_debug.py --root .\\dataset
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

P = argparse.ArgumentParser()
P.add_argument("--root", default="dataset")
a = P.parse_args()

root = Path(a.root).resolve()
cache = root.parent / "cache"
from phase3_baseline import load_split

G = np.load(cache / "gallery.npz", allow_pickle=True)
Q = np.load(cache / "query.npz", allow_pickle=True)
C = np.load(cache / "captions.npz", allow_pickle=True)

sp = load_split(root / "PRW", "frame_test.mat")
mask = np.array([f in sp for f in G["frames"]]) if sp else np.ones(len(G["frames"]), bool)
gp, gc = G["ids"][mask], G["cams"][mask]
q_pid, q_cam, q_frames = Q["ids"], Q["cams"], Q["frames"]

print(f"gallery crops        {len(gp)}")
print(f"gallery cameras      {sorted(set(gc.tolist()))}")
print(f"gallery identities   {len(set(gp[gp > 0].tolist()))}")
print(f"query boxes          {len(q_pid)}")

key = {(int(p), str(f)): i for i, (p, f) in enumerate(zip(q_pid, q_frames))}
caps_of = {}
for i, (p, f) in enumerate(zip(C["ids"], C["frames"])):
    r = key.get((int(p), str(f)))
    if r is not None:
        caps_of.setdefault(r, []).append(i)
print(f"queries with captions {len(caps_of)}")

by_id = {}
for i in range(len(q_pid)):
    if i in caps_of:
        by_id.setdefault(int(q_pid[i]), []).append(i)
sizes = [len(v) for v in by_id.values()]
print(f"query boxes per identity: min {min(sizes)} median "
      f"{int(np.median(sizes))} max {max(sizes)}")

print("\nfirst 10 queries, with the capped 2-photograph group:")
print(f"  {'q':>4}{'pid':>6}{'cam':>5}{'group':>18}{'excl':>10}"
      f"{'gallery left':>14}{'relevant':>10}")
print("  " + "-" * 70)
ok = 0
for qi in list(caps_of)[:10]:
    pid = int(q_pid[qi])
    group = [qi] + [j for j in by_id[pid] if j != qi][:1]
    excl = set(q_cam[group].tolist())
    idx = np.flatnonzero(~np.isin(gc, list(excl)))
    rel = (gp[idx] == pid) if len(idx) else np.zeros(0, bool)
    ok += int(rel.any())
    print(f"  {qi:>4}{pid:>6}{q_cam[qi]:>5}"
          f"{str(sorted(q_cam[group].tolist())):>18}{str(sorted(excl)):>10}"
          f"{len(idx):>14}{int(rel.sum()):>10}")

print(f"\n{ok} of 10 scorable")
if ok == 0:
    print("\nNo query has a relevant gallery item after excluding its own")
    print("cameras. Check that the gallery identities and the query")
    print("identities come from the same numbering.")
    inter = set(gp.tolist()) & set(q_pid.tolist())
    print(f"identities shared between gallery and queries: {len(inter)}")
else:
    print("\nThe protocol is fine. If phase8_full.py still reports NaN, the")
    print("file on disk is the old version - re-save it.")
