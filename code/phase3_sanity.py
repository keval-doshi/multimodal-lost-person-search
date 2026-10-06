"""
Phase 3a - Backbone sanity gate.

Before encoding 66,557 PRW crops, prove the appearance model is wired
correctly by evaluating it on Market-1501 under the standard protocol.

If this gate fails, every number produced later is meaningless, so nothing
else runs until it passes.

  python phase3_sanity.py --root .\\dataset

Design notes
------------
The checkpoint used is trained on MSMT17, not on Market-1501, so this is a
CROSS-DOMAIN evaluation and will score well below the ~94.8% Rank-1 reported
for same-domain OSNet. That is deliberate:

  * The PRW readme states PRW and Market-1501 share the same origin - same
    campus, same cameras. A Market-1501-trained backbone applied to PRW
    would be close to in-domain, which is hard to defend.
  * MSMT17 is a different campus entirely, so the backbone is cleanly
    cross-domain to both benchmarks used in this project.

The gate therefore checks that the model works, not that it matches a
same-domain published figure. Random features would give roughly 0.1%
Rank-1 on 750 identities; anything above 40% proves the weights loaded and
the preprocessing is right.

Market-1501 protocol
--------------------
Filenames are pid_camid_sequence_frame.jpg, e.g. 0002_c1s1_000451_03.jpg.
  pid = -1  distractor, always ignored
  pid =  0  background, ignored in the gallery
For each query, gallery images sharing BOTH the identity and the camera are
junk and are removed from the ranking, since matching a person to another
shot from the same camera is trivial.
"""

import argparse
import importlib.util
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

OSNET_URL = ("https://raw.githubusercontent.com/KaiyangZhou/"
             "deep-person-reid/master/torchreid/models/osnet.py")
HF_REPO = "kaiyangzhou/osnet"
HF_FILE = ("osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_"
           "lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")

GATE_RANK1 = 40.0          # below this, assume something is wired wrong

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ------------------------------------------------------------- model
def get_osnet_module(workdir: Path):
    """Download the architecture file once, then import it directly."""
    local = workdir / "osnet_arch.py"
    if not local.exists():
        print("Fetching OSNet architecture ...")
        try:
            urllib.request.urlretrieve(OSNET_URL, local)
        except Exception as e:
            sys.exit(f"ERROR: could not download OSNet source ({e}).\n"
                     f"Download {OSNET_URL} manually and save it as {local}")
    spec = importlib.util.spec_from_file_location("osnet_arch", local)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_weights(workdir: Path, path_override: str):
    """
    Download the checkpoint directly to a short filename.

    huggingface_hub's cache builds a path like
        <dir>/models--kaiyangzhou--osnet/snapshots/<40-char hash>/<100-char name>
    which overruns the Windows 260-character path limit when the project sits
    in a deep folder. Fetching the URL straight to a short name avoids it.
    """
    if path_override:
        p = Path(path_override)
        if not p.exists():
            sys.exit(f"ERROR: {p} not found")
        return p

    local = workdir / "osnet_x1_0_msmt17.pth"
    if local.exists() and local.stat().st_size > 1_000_000:
        print(f"Weights already present: {local.name} "
              f"({local.stat().st_size / 1e6:.1f} MB)")
        return local

    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{HF_FILE}?download=true"
    print(f"Downloading weights -> {local.name}")

    def hook(blocks, bs, total):
        if total > 0:
            pct = min(100, blocks * bs * 100 // total)
            print(f"\r  {pct:3d}%  ({total / 1e6:.1f} MB)", end="", flush=True)

    tmp = workdir / "osnet.part"
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=hook)
        print()
    except Exception as e:
        tmp.unlink(missing_ok=True)
        sys.exit(f"\nERROR: download failed ({e}).\n"
                 f"Download this manually in a browser:\n  {url}\n"
                 f"Save it as: {local}\n"
                 f"Then rerun, or pass --weights <path>")
    tmp.replace(local)
    print(f"  saved {local.stat().st_size / 1e6:.1f} MB")
    return local


def build_model(workdir, weights_path, device):
    mod = get_osnet_module(workdir)
    model = mod.osnet_x1_0(num_classes=1000, pretrained=False, loss="softmax")

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

    msd = model.state_dict()
    matched = {k: v for k, v in sd.items()
               if k in msd and msd[k].shape == v.shape}
    missing = [k for k in msd if k not in matched]
    model.load_state_dict(matched, strict=False)

    print(f"  loaded {len(matched)}/{len(msd)} tensors")
    # only the classifier head should be unmatched - it is unused at inference
    non_cls = [k for k in missing if not k.startswith("classifier")]
    if non_cls:
        print(f"  WARNING: {len(non_cls)} non-classifier tensors did not load,")
        print(f"           e.g. {non_cls[:3]}")
        print("           the architecture may not match the checkpoint")
    elif missing:
        print("  (classifier head unmatched, as expected - not used for features)")

    return model.to(device).eval()


# ------------------------------------------------------------- data
def parse_name(fname):
    """0002_c1s1_000451_03.jpg -> (pid, camid)"""
    parts = fname.split("_")
    pid = int(parts[0])
    cam = int(parts[1][1])
    return pid, cam


def list_split(folder: Path):
    items = []
    for f in sorted(folder.glob("*.jpg")):
        try:
            pid, cam = parse_name(f.name)
        except (ValueError, IndexError):
            continue
        items.append((f, pid, cam))
    return items


@torch.no_grad()
def encode(model, items, device, batch=128, size=(256, 128), label=""):
    import cv2
    feats = np.zeros((len(items), 512), np.float32)
    buf, idx = [], []
    t0 = time.time()

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            f = model(x)
        f = torch.nn.functional.normalize(f.float(), dim=1)
        feats[idx] = f.cpu().numpy()
        buf.clear(); idx.clear()

    for i, (p, _, _) in enumerate(items):
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        im = (im - MEAN) / STD
        buf.append(im.transpose(2, 0, 1))
        idx.append(i)
        if len(buf) == batch:
            flush()
        if (i + 1) % 4000 == 0:
            print(f"    {label} {i + 1}/{len(items)}  "
                  f"{(i + 1) / (time.time() - t0):.0f} img/s")
    flush()
    return feats


# ------------------------------------------------------------- eval
def market_eval(qf, q_pid, q_cam, gf, g_pid, g_cam, topk=(1, 5, 10)):
    """Standard Market-1501 CMC and mAP with junk removal."""
    sim = qf @ gf.T                       # both L2-normalised
    order = np.argsort(-sim, axis=1)

    all_cmc, all_ap = [], []
    max_k = max(topk)
    for i in range(len(qf)):
        o = order[i]
        keep = ~((g_pid[o] == q_pid[i]) & (g_cam[o] == q_cam[i]))
        keep &= (g_pid[o] != -1)
        o = o[keep]

        match = (g_pid[o] == q_pid[i]).astype(np.int32)
        if match.sum() == 0:
            continue

        cmc = match.cumsum()
        cmc[cmc > 1] = 1
        # pad short rankings so every row is the same length
        if len(cmc) < max_k:
            cmc = np.concatenate([cmc, np.full(max_k - len(cmc), cmc[-1])])
        all_cmc.append(cmc[:max_k])

        pos = np.where(match == 1)[0]
        prec = [(j + 1) / (p + 1) for j, p in enumerate(pos)]
        all_ap.append(float(np.mean(prec)))

    cmc = np.stack(all_cmc).mean(0) * 100
    return {f"Rank-{k}": float(cmc[k - 1]) for k in topk} | \
           {"mAP": float(np.mean(all_ap) * 100), "queries": len(all_ap)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--weights", default="", help="local .pth instead of HF")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit-gallery", type=int, default=0,
                    help="subsample the gallery for a fast check")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    work = root.parent / "models"
    work.mkdir(exist_ok=True)

    mk = root / "Market-1501"
    if not (mk / "query").is_dir():
        nested = next((p for p in mk.glob("Market-1501*")
                       if (p / "query").is_dir()), None)
        if nested is None:
            sys.exit(f"ERROR: Market-1501 query folder not found under {mk}")
        mk = nested
    print(f"Market-1501: {mk.name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected, this will be slow")

    model = build_model(work, get_weights(work, a.weights), device)

    q = list_split(mk / "query")
    g = list_split(mk / "bounding_box_test")
    if a.limit_gallery:
        rng = np.random.default_rng(0)
        pick = sorted(rng.choice(len(g), min(a.limit_gallery, len(g)), replace=False))
        g = [g[i] for i in pick]
    print(f"\nQuery {len(q)}  Gallery {len(g)}")

    print("Encoding ...")
    qf = encode(model, q, device, a.batch, label="query")
    gf = encode(model, g, device, a.batch, label="gallery")

    q_pid = np.array([p for _, p, _ in q]); q_cam = np.array([c for _, _, c in q])
    g_pid = np.array([p for _, p, _ in g]); g_cam = np.array([c for _, _, c in g])

    r = market_eval(qf, q_pid, q_cam, gf, g_pid, g_cam)

    print("\n" + "=" * 58)
    print("MARKET-1501  (cross-domain: backbone trained on MSMT17)")
    print("=" * 58)
    for k in ("Rank-1", "Rank-5", "Rank-10", "mAP"):
        print(f"  {k:<9}{r[k]:6.1f} %")
    print(f"  queries scored: {r['queries']}")
    print(f"  identities in gallery: {len(set(g_pid.tolist()) - {-1, 0})}")

    print("\n" + "-" * 58)
    if r["Rank-1"] >= GATE_RANK1:
        print(f"GATE PASSED  (Rank-1 {r['Rank-1']:.1f}% >= {GATE_RANK1}%)")
        print("Weights loaded and preprocessing is correct.")
        print("Note this is cross-domain; same-domain OSNet reports ~94.8%")
        print("Rank-1 on Market-1501. The gap is expected and should be")
        print("stated in the methodology.")
        print("\nProceed to Phase 3b (encode the PRW gallery).")
    else:
        print(f"GATE FAILED  (Rank-1 {r['Rank-1']:.1f}% < {GATE_RANK1}%)")
        print("Random features would score about 0.1%, so the model is doing")
        print("something - but not enough. Check, in order:")
        print("  1. did all non-classifier tensors load above?")
        print("  2. is the input 256x128 (height x width), not 128x256?")
        print("  3. is the image RGB, not BGR?")
        print("  4. are ImageNet mean/std applied?")
        print("\nDo NOT proceed to encoding until this passes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
