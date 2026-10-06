"""
Phase 2a - Person detection over PRW frames.

Runs YOLO in two modes and caches every detection to disk:

  full    - one forward pass over the whole 1920x1080 frame
  tiled   - overlapping tiles (SAHI-style slicing aided inference,
            Akyon et al. 2022) plus an optional whole-frame pass,
            merged with NMS in original frame coordinates

The tiled mode is what should recover small, distant people. Comparing the
two is the O2 result.

Typical use:

  # smoke test first - 40 frames, both modes, ~1 minute
  python phase2_detect.py --root .\\dataset --limit 40 --mode both

  # then the real run
  python phase2_detect.py --root .\\dataset --mode full
  python phase2_detect.py --root .\\dataset --mode tiled

Output: dataset/../cache/det_full.parquet, cache/det_tiled.parquet
Both are resumable - rerun after an interruption and it continues.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------- setup
def banner(msg):
    print(f"\n{'=' * 62}\n{msg}\n{'=' * 62}")


def check_device(allow_cpu):
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name}  ({vram:.1f} GB)")
        return "cuda"
    if allow_cpu:
        print("WARNING: running on CPU. This will take many hours.")
        return "cpu"
    print("ERROR: CUDA not available. Detection on CPU takes 10+ hours.")
    print("       Check you are using the Python of your virtual environment")
    print("       and that PyTorch was installed with CUDA support.")
    print("       Pass --allow-cpu to override.")
    sys.exit(1)


# ---------------------------------------------------------------- tiling
def tile_positions(size, tile, overlap):
    """Start coordinates covering `size` with `tile`-wide windows."""
    if size <= tile:
        return [0]
    step = int(tile * (1 - overlap))
    pos = list(range(0, size - tile + 1, step))
    if pos[-1] + tile < size:          # ensure the far edge is covered
        pos.append(size - tile)
    return pos


def nms(boxes, scores, iou_thr):
    """Greedy NMS. boxes: (N,4) xyxy. Returns kept indices, pure NumPy."""
    if len(boxes) == 0:
        return np.empty(0, dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[iou < iou_thr]
    return np.array(keep, dtype=int)


def suppress_contained(boxes, scores, thr=0.75):
    """
    Drop boxes that sit almost entirely inside a higher-scoring box.

    Plain NMS uses IoU, which is small when a tile fragment (say, a torso)
    is compared with the full person box - so fragments survive. Measuring
    overlap against the *smaller* box's own area catches them.
    """
    if len(boxes) < 2:
        return np.arange(len(boxes))
    order = scores.argsort()[::-1]
    b = boxes[order]
    area = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    keep = []
    for i in range(len(b)):
        drop = False
        for j in keep:
            ix1 = max(b[i, 0], b[j, 0]); iy1 = max(b[i, 1], b[j, 1])
            ix2 = min(b[i, 2], b[j, 2]); iy2 = min(b[i, 3], b[j, 3])
            inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
            if inter / (min(area[i], area[j]) + 1e-9) > thr:
                drop = True
                break
        if not drop:
            keep.append(i)
    return order[np.array(keep, dtype=int)]


def detect_tiled(model, img, tile, overlap, conf, iou_merge,
                 include_full, imgsz, contain_thr=0.75, edge_margin=4):
    """Run the model over overlapping tiles and merge back to frame coords."""
    h, w = img.shape[:2]
    xs = tile_positions(w, tile, overlap)
    ys = tile_positions(h, tile, overlap)

    crops, offsets = [], []
    for y0 in ys:
        for x0 in xs:
            crops.append(img[y0:y0 + tile, x0:x0 + tile])
            offsets.append((x0, y0))

    all_b, all_s = [], []

    # tiles - one batched forward pass
    res = model.predict(crops, imgsz=tile, conf=conf, classes=[0],
                        verbose=False, device=model.device)
    for r, (x0, y0) in zip(res, offsets):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        b = r.boxes.xyxy.cpu().numpy()
        sc = r.boxes.conf.cpu().numpy()

        # A box touching a tile edge is probably a person cut in half by the
        # slice. Drop it unless that edge is also the frame edge, where the
        # truncation is real.
        if edge_margin > 0:
            m = edge_margin
            cut = np.zeros(len(b), bool)
            if x0 > 0:
                cut |= b[:, 0] < m
            if y0 > 0:
                cut |= b[:, 1] < m
            if x0 + tile < w:
                cut |= b[:, 2] > tile - m
            if y0 + tile < h:
                cut |= b[:, 3] > tile - m
            b, sc = b[~cut], sc[~cut]
            if len(b) == 0:
                continue

        b[:, [0, 2]] += x0
        b[:, [1, 3]] += y0
        all_b.append(b)
        all_s.append(sc)

    # whole frame as well - SAHI's "hyper inference" keeps large objects
    if include_full:
        r = model.predict(img, imgsz=imgsz, conf=conf, classes=[0],
                          verbose=False, device=model.device)[0]
        if r.boxes is not None and len(r.boxes):
            all_b.append(r.boxes.xyxy.cpu().numpy())
            all_s.append(r.boxes.conf.cpu().numpy())

    if not all_b:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32)

    boxes = np.concatenate(all_b).astype(np.float32)
    scores = np.concatenate(all_s).astype(np.float32)
    keep = nms(boxes, scores, iou_merge)
    boxes, scores = boxes[keep], scores[keep]
    keep = suppress_contained(boxes, scores, contain_thr)
    return boxes[keep], scores[keep]


# ---------------------------------------------------------------- main
def run(args):
    import cv2
    from ultralytics import YOLO

    root = Path(args.root).resolve()
    frames_dir = root / "PRW" / "frames"
    if not frames_dir.is_dir():
        sys.exit(f"ERROR: {frames_dir} not found. Check --root.")

    cache = root.parent / "cache"
    cache.mkdir(exist_ok=True)

    device = check_device(args.allow_cpu)
    print(f"Loading {args.model} ...")
    model = YOLO(args.model)
    model.to(device)

    frames = sorted(f.name for f in frames_dir.glob("*.jpg"))

    if args.frame_list:
        fl = Path(args.frame_list)
        if not fl.is_absolute():
            fl = (Path.cwd() / fl).resolve()
        if not fl.exists():
            sys.exit(f"ERROR: frame list {fl} not found")
        want = {l.strip() for l in fl.read_text(encoding="utf-8").splitlines() if l.strip()}
        frames = [f for f in frames if f in want]
        print(f"Frame list {fl.name}: {len(frames)} frames selected")

    if args.limit:
        # sample across the whole dataset, not the first N alphabetically -
        # the first frames are all one camera and one segment
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(frames), size=min(args.limit, len(frames)),
                          replace=False)
        frames = [frames[i] for i in sorted(pick)]
        cams = sorted({f.split("s")[0] for f in frames})
        print(f"Random sample of {len(frames)} frames, seed {args.seed}, "
              f"cameras: {', '.join(cams)}")
    print(f"Frames to process: {len(frames)}")

    modes = ["full", "tiled"] if args.mode == "both" else [args.mode]

    for mode in modes:
        tag = args.out_suffix or ("_smoke" if args.limit else "")
        out = cache / f"det_{mode}{tag}.parquet"
        part = out.with_suffix(".partial.parquet")

        done = set()
        rows = []
        if part.exists() and not args.restart:
            prev = pd.read_parquet(part)
            rows = prev.to_dict("records")
            done = set(prev["frame"].unique())
            print(f"Resuming {mode}: {len(done)} frames already done")

        todo = [f for f in frames if f not in done]
        banner(f"MODE: {mode}   frames remaining: {len(todo)}")
        if not todo:
            print("nothing to do")
        t0 = time.time()

        for i, fname in enumerate(todo, 1):
            img = cv2.imread(str(frames_dir / fname))
            if img is None:
                print(f"  WARN unreadable: {fname}")
                continue

            if mode == "full":
                r = model.predict(img, imgsz=args.imgsz, conf=args.conf,
                                  classes=[0], verbose=False, device=device)[0]
                if r.boxes is not None and len(r.boxes):
                    b = r.boxes.xyxy.cpu().numpy()
                    s = r.boxes.conf.cpu().numpy()
                else:
                    b, s = np.empty((0, 4)), np.empty(0)
            else:
                b, s = detect_tiled(model, img, args.tile, args.overlap,
                                    args.conf, args.iou_merge,
                                    not args.no_full_fusion, args.imgsz,
                                    args.contain_thr, args.edge_margin)

            for (x1, y1, x2, y2), sc in zip(b, s):
                rows.append({"frame": fname,
                             "x1": float(x1), "y1": float(y1),
                             "x2": float(x2), "y2": float(y2),
                             "conf": float(sc)})

            if i % args.save_every == 0 or i == len(todo):
                pd.DataFrame(rows).to_parquet(part, index=False)
                el = time.time() - t0
                rate = i / el
                eta = (len(todo) - i) / rate / 60
                print(f"  {i}/{len(todo)}  {rate:.1f} fps  "
                      f"ETA {eta:.0f} min  dets so far {len(rows)}")

        df = pd.DataFrame(rows)
        df.to_parquet(out, index=False)
        if part.exists():
            part.unlink()

        n_f = df["frame"].nunique() if len(df) else 0
        print(f"\nSaved {out}")
        print(f"  {len(df)} detections over {n_f} frames "
              f"({len(df) / max(n_f, 1):.1f} per frame)")
        if len(df):
            hh = df["y2"] - df["y1"]
            print(f"  box height  median {hh.median():.0f} px   "
                  f"p10 {hh.quantile(.1):.0f}   p90 {hh.quantile(.9):.0f}")
            print(f"  under 64 px: {(hh < 64).mean() * 100:.1f}% of detections")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset", help="folder containing PRW/")
    ap.add_argument("--mode", default="both", choices=["full", "tiled", "both"])
    ap.add_argument("--model", default="yolo11m.pt",
                    help="downloaded automatically on first use")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="inference size for whole-frame passes")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="keep low - recall matters more than precision here")
    ap.add_argument("--iou-merge", type=float, default=0.55,
                    help="NMS IoU when merging tile predictions")
    ap.add_argument("--no-full-fusion", action="store_true",
                    help="tiled mode: skip the extra whole-frame pass")
    ap.add_argument("--frame-list", default="",
                    help="text file of frame names, one per line; "
                         "restrict processing to these")
    ap.add_argument("--out-suffix", default="",
                    help="appended to the cache filename, e.g. _small")
    ap.add_argument("--limit", type=int, default=0,
                    help="random sample of N frames (smoke test)")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed - keep fixed for reproducibility")
    ap.add_argument("--contain-thr", type=float, default=0.75,
                    help="drop a box if this fraction of it sits inside a "
                         "higher-scoring box")
    ap.add_argument("--edge-margin", type=int, default=4,
                    help="discard tile detections touching an interior tile "
                         "edge; 0 disables")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--restart", action="store_true",
                    help="ignore any partial cache and start over")
    ap.add_argument("--allow-cpu", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
