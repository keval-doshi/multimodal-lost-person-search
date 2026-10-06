"""
Phase 3c - Encode the PRW gallery, the queries and the captions.

Produces every vector the rest of the project needs. Run once; Phases 4, 5
and 6 all read the caches this writes.

  # smoke test: 300 frames, about a minute
  python phase3_encode.py --root .\\dataset --limit 300

  # full run
  python phase3_encode.py --root .\\dataset

Three encoders, one pass over the frames:
  OSNet          appearance embedding of each crop      (512-d)
  IRRA image     same crop in the caption-aligned space (512-d)
  IRRA text      each caption                           (512-d)

Identity assignment
-------------------
A detected box has no identity of its own, so each is matched to the PRW
ground truth by overlap. Detections matching nothing are kept with id -1:
they are the distractors a real search has to rank past, and discarding them
would make the task artificially easy.

Outputs (in cache/)
  gallery.npz    features, ids, cameras, frames, boxes, heights, confidence
  query.npz      the 2,057 hand-drawn query boxes, both encoders
  captions.npz   4,114 caption embeddings with their identities
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ----------------------------------------------------------- OSNet
def load_osnet(models_dir: Path, device):
    import importlib.util
    import urllib.request

    arch = models_dir / "osnet_arch.py"
    if not arch.exists():
        url = ("https://raw.githubusercontent.com/KaiyangZhou/"
               "deep-person-reid/master/torchreid/models/osnet.py")
        print("Fetching OSNet architecture ...")
        urllib.request.urlretrieve(url, arch)
    spec = importlib.util.spec_from_file_location("osnet_arch", arch)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    w = models_dir / "osnet_x1_0_msmt17.pth"
    if not w.exists():
        sys.exit(f"ERROR: {w} not found. Run phase3_sanity.py first.")

    model = mod.osnet_x1_0(num_classes=1000, pretrained=False, loss="softmax")
    ck = torch.load(w, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    msd = model.state_dict()
    matched = {k: v for k, v in sd.items()
               if k in msd and msd[k].shape == v.shape}
    model.load_state_dict(matched, strict=False)
    print(f"OSNet: {len(matched)}/{len(msd)} tensors")
    return model.to(device).eval()


def prep_osnet(bgr):
    import cv2
    im = cv2.resize(bgr, (128, 256), interpolation=cv2.INTER_CUBIC)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((im - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


@torch.no_grad()
def encode_osnet(model, arrays, device, batch=128):
    out = []
    for i in range(0, len(arrays), batch):
        x = torch.from_numpy(np.stack(arrays[i:i + batch])).to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            f = model(x)
        f = torch.nn.functional.normalize(f.float(), dim=1)
        out.append(f.cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 512), np.float32)


# ----------------------------------------------------------- ground truth
def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    ar_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ar_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (ar_a[:, None] + ar_b[None, :] - inter + 1e-9)


def load_gt_boxes(prw: Path, frames):
    """frame -> (boxes xyxy, ids). Column order is detected from the data."""
    from scipy.io import loadmat

    def first(m):
        for k, v in m.items():
            if not k.startswith("__") and isinstance(v, np.ndarray) and v.size:
                return v
        return None

    ann = prw / "annotations"
    sample = []
    for f in frames[:50]:
        p = ann / (f + ".mat")
        if p.exists():
            a = first(loadmat(str(p)))
            if a is not None and a.ndim == 2 and a.shape[1] >= 5:
                sample.append(np.atleast_2d(a)[:, :5])
    stack = np.vstack(sample).astype(float)

    def geom(x):
        w, h = x[:, 2], x[:, 3]
        return ((w > 0) & (h > 0) & (x[:, 0] + w <= 1922)
                & (x[:, 1] + h <= 1082)).mean()

    id_first = geom(stack[:, 1:5]) >= geom(stack[:, 0:4])
    print(f"Annotation layout: {'id_first' if id_first else 'id_last'}")

    gt = {}
    for f in frames:
        p = ann / (f + ".mat")
        if not p.exists():
            gt[f] = (np.empty((0, 4)), np.empty(0, int))
            continue
        a = first(loadmat(str(p)))
        if a is None or a.ndim != 2 or a.shape[1] < 5:
            gt[f] = (np.empty((0, 4)), np.empty(0, int))
            continue
        a = np.atleast_2d(a).astype(float)
        xywh = a[:, 1:5] if id_first else a[:, 0:4]
        ids = (a[:, 0] if id_first else a[:, 4]).astype(int)
        x, y, w, h = xywh.T
        gt[f] = (np.stack([x, y, x + w, y + h], 1), ids)
    return gt


def camera_of(frame_name):
    return int(frame_name.split("s")[0][1:])


# ----------------------------------------------------------- main
def run(a):
    import cv2

    root = Path(a.root).resolve()
    prw = root / "PRW"
    cache = root.parent / "cache"
    models = root.parent / "models"
    cache.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu" and not a.allow_cpu:
        sys.exit("ERROR: CUDA not available. Pass --allow-cpu to override.")

    det_path = cache / "det_full.parquet"
    if not det_path.exists():
        sys.exit(f"ERROR: {det_path} not found. Run Phase 2 first.")
    det = pd.read_parquet(det_path)
    print(f"Detections: {len(det)}")
    det = det[det["conf"] >= a.min_conf].reset_index(drop=True)
    print(f"  after conf >= {a.min_conf}: {len(det)}")

    frames = sorted(det["frame"].unique())
    if a.limit:
        frames = frames[:a.limit]
        det = det[det["frame"].isin(frames)]
        print(f"  smoke test: {len(frames)} frames, {len(det)} detections")

    print("\nLoading models ...")
    osnet = load_osnet(models, device)
    from phase3_irra import IRRAEncoder
    irra = IRRAEncoder(a.repo, a.ckpt, device)

    print("\nLoading ground truth ...")
    gt = load_gt_boxes(prw, frames)

    # ---------------- gallery ----------------
    print(f"\nEncoding gallery over {len(frames)} frames ...")
    by_frame = {f: g for f, g in det.groupby("frame")}
    # Preprocessed crops are ~1 MB each across the two input formats, so they
    # are encoded in chunks and only the 512-d vectors are retained (2 KB each).
    # Holding them all would need roughly 50 GB.
    buf_os, buf_ir = [], []
    feat_os, feat_ir = [], []
    G = {k: [] for k in ("ids", "cams", "frames", "boxes", "conf")}
    t0, n_assigned = time.time(), 0

    def flush():
        if not buf_os:
            return
        feat_os.append(encode_osnet(osnet, buf_os, device, a.batch))
        feat_ir.append(irra.encode_images(buf_ir, max(a.batch // 4, 16)))
        buf_os.clear(); buf_ir.clear()

    for i, fname in enumerate(frames, 1):
        img = cv2.imread(str(prw / "frames" / fname))
        if img is None:
            continue
        d = by_frame[fname]
        boxes = d[["x1", "y1", "x2", "y2"]].to_numpy()
        confs = d["conf"].to_numpy()

        gb, gi = gt.get(fname, (np.empty((0, 4)), np.empty(0, int)))
        ids = np.full(len(boxes), -1, int)
        if len(gb):
            M = iou_matrix(boxes, gb)
            taken = np.zeros(len(gb), bool)
            for di in np.argsort(-confs):
                gj = int(np.argmax(M[di]))
                if M[di, gj] >= a.iou and not taken[gj]:
                    ids[di] = gi[gj]
                    taken[gj] = True
        n_assigned += int((ids > 0).sum())

        cam = camera_of(fname)
        H, W = img.shape[:2]
        for (x1, y1, x2, y2), c, pid in zip(boxes, confs, ids):
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(W, int(x2)), min(H, int(y2))
            if x2 - x1 < 8 or y2 - y1 < 16:
                continue
            crop = img[y1:y2, x1:x2]
            buf_os.append(prep_osnet(crop))
            buf_ir.append(irra.preprocess(crop))
            G["ids"].append(pid); G["cams"].append(cam)
            G["frames"].append(fname); G["boxes"].append([x1, y1, x2, y2])
            G["conf"].append(c)

        if len(buf_os) >= a.chunk:
            flush()

        if i % 500 == 0 or i == len(frames):
            el = time.time() - t0
            done = sum(len(f) for f in feat_os) + len(buf_os)
            print(f"  {i}/{len(frames)} frames  {done} crops  "
                  f"{i / el:.1f} fps  ETA {(len(frames) - i) / (i / el) / 60:.0f} min")

    flush()
    print(f"\nCrops: {len(G['ids'])}   with a known identity: {n_assigned} "
          f"({n_assigned / max(len(G['ids']), 1) * 100:.1f}%)")

    g_os = np.concatenate(feat_os) if feat_os else np.zeros((0, 512), np.float32)
    g_ir = np.concatenate(feat_ir) if feat_ir else np.zeros((0, 512), np.float32)
    assert len(g_os) == len(G["ids"]) == len(g_ir), (
        f"feature/metadata mismatch: {len(g_os)} {len(g_ir)} {len(G['ids'])}")

    boxes = np.array(G["boxes"], np.int32)
    np.savez_compressed(
        cache / f"gallery{'_smoke' if a.limit else ''}.npz",
        osnet=g_os, irra=g_ir,
        ids=np.array(G["ids"], np.int32), cams=np.array(G["cams"], np.int32),
        frames=np.array(G["frames"]), boxes=boxes,
        conf=np.array(G["conf"], np.float32),
        height=(boxes[:, 3] - boxes[:, 1]).astype(np.int32))
    print(f"  gallery: osnet {g_os.shape}  irra {g_ir.shape}")

    if a.limit:
        print("\nSmoke test complete. Rerun without --limit for the full set.")
        return

    # ---------------- queries ----------------
    print("\nEncoding query boxes ...")
    qdir = prw / "query_box"
    q_os_in, q_ir_in, q_ids, q_cams, q_frames = [], [], [], [], []
    for p in sorted(qdir.glob("*.jpg")):
        pid, _, rest = p.stem.partition("_")
        img = cv2.imread(str(p))
        if img is None:
            continue
        q_os_in.append(prep_osnet(img)); q_ir_in.append(irra.preprocess(img))
        q_ids.append(int(pid)); q_frames.append(rest + ".jpg")
        q_cams.append(camera_of(rest))
    q_os = encode_osnet(osnet, q_os_in, device, a.batch)
    q_ir = irra.encode_images(q_ir_in, max(a.batch // 4, 16))
    np.savez_compressed(cache / "query.npz", osnet=q_os, irra=q_ir,
                        ids=np.array(q_ids, np.int32),
                        cams=np.array(q_cams, np.int32),
                        frames=np.array(q_frames))
    print(f"  queries: {len(q_ids)}  identities {len(set(q_ids))}")

    # ---------------- captions ----------------
    print("\nEncoding captions ...")
    import json
    tb = next((p for p in (prw / "PRW-TBPS_test", root / "PRW-TBPS" /
                           "PRW-TBPS_test") if p.exists()), None)
    if tb is None:
        hits = list(root.rglob("PRW-TBPS_test"))
        tb = hits[0] if hits else None
    if tb is None:
        print("  WARNING: PRW-TBPS_test not found, skipping captions")
        return
    recs = json.load(open(tb, encoding="utf-8"))
    key = "desrciption" if "desrciption" in recs[0] else "description"

    caps, c_ids, c_frames, c_slot = [], [], [], []
    for r in recs:
        for j, c in enumerate(r[key]):
            c = c.strip()
            if c:
                caps.append(c); c_ids.append(r["id"])
                c_frames.append(r["pic_path"]); c_slot.append(j)
    tf = irra.encode_text(caps, batch=256)
    np.savez_compressed(cache / "captions.npz", irra=tf,
                        ids=np.array(c_ids, np.int32),
                        frames=np.array(c_frames),
                        slot=np.array(c_slot, np.int32),
                        text=np.array(caps))
    print(f"  captions: {tf.shape}  identities {len(set(c_ids))}")

    print("\n" + "=" * 58)
    print("ENCODING COMPLETE")
    print(f"  gallery  {g_os.shape[0]:>7} crops")
    print(f"  queries  {len(q_ids):>7}")
    print(f"  captions {tf.shape[0]:>7}")
    print("Proceed to phase3_baseline.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--repo", default="IRRA")
    ap.add_argument("--ckpt", default="models/irra/best.pth")
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="drop weak detections before encoding")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="overlap needed to inherit a ground-truth identity")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=2000,
                    help="crops buffered before encoding; lower this if RAM "
                         "is tight (2000 needs about 2 GB)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-cpu", action="store_true")
    run(ap.parse_args())
