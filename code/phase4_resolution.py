"""
Phase 4 - Resolution study.

Degrades the gallery crops through a ladder of target heights and repeats the
photograph-only and description-only baselines at each level. This is the
direct answer to RQ3.

  # smoke test: 400 frames, a couple of minutes
  python phase4_resolution.py --root .\\dataset --limit 400

  # full run
  python phase4_resolution.py --root .\\dataset

Protocol
--------
Only the GALLERY is degraded. The query photograph and the descriptions stay
at full quality, because in a real search you have a good recent photograph
of the missing person and it is the camera footage that is distant.

Degradation: a crop taller than the target is resampled down to the target
height, preserving aspect, then resampled back up to the model input size.
This is the standard multi-level-resolution protocol. A crop already shorter
than the target is left alone, and the proportion actually degraded is
reported at every level, since PRW crops are not uniformly large.

The ladder starts below the native size distribution on purpose. Phase 2
established a median ground-truth height of 163 px and a 25th percentile of
122 px, so a level at 128 px would upsample a large share of crops rather
than degrade them.

Frames are read once and all levels are produced from the same crop, so the
expensive part - decoding 1920x1080 JPEGs - happens a single time.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

LEVELS = [0, 96, 64, 48, 32, 24, 16]        # 0 = native, no degradation
LEVEL_NAME = {0: "native"}


def lname(h):
    return LEVEL_NAME.get(h, f"{h}px")


def evaluate_hits(qf, q_pid, q_cam, gf, g_pid, g_cam, chunk=512):
    """Per-query Rank-1 hit vector, for bootstrap confidence intervals."""
    hits = []
    for s in range(0, len(qf), chunk):
        e = min(s + chunk, len(qf))
        sim = qf[s:e] @ gf.T
        order = np.argsort(-sim, axis=1)
        for r in range(e - s):
            i = s + r
            o = order[r][~(g_cam[order[r]] == q_cam[i])]
            match = (g_pid[o] == q_pid[i])
            if match.sum() == 0:
                continue
            hits.append(bool(match[0]))
    return np.array(hits, bool)


def degrade(bgr, target_h):
    """Downsample to target height, then back up. Returns the crop unchanged
    if it is already at or below the target."""
    import cv2
    if target_h == 0:
        return bgr
    h, w = bgr.shape[:2]
    if h <= target_h:
        return bgr
    tw = max(2, int(round(w * target_h / h)))
    small = cv2.resize(bgr, (tw, target_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def main():
    import cv2
    from phase3_encode import load_osnet, prep_osnet, encode_osnet, camera_of
    from phase3_irra import IRRAEncoder
    from phase3_baseline import evaluate, load_split

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--repo", default="IRRA")
    ap.add_argument("--ckpt", default="models/irra/best.pth")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=256,
                    help="crops buffered per level before encoding; "
                         "256 needs about 1.5 GB across the whole ladder")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    root = Path(a.root).resolve()
    prw = root / "PRW"
    cache = root.parent / "cache"
    models = root.parent / "models"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    G = np.load(cache / "gallery.npz", allow_pickle=True)
    Q = np.load(cache / "query.npz", allow_pickle=True)
    C = np.load(cache / "captions.npz", allow_pickle=True)

    # restrict to the official test split, as in Phase 3
    sp = load_split(prw, "frame_test.mat")
    g_frames_all = G["frames"]
    mask = np.array([f in sp for f in g_frames_all]) if sp else np.ones(len(g_frames_all), bool)
    print(f"Gallery: {mask.sum()} of {len(mask)} crops on test frames")

    meta = pd.DataFrame({
        "frame": g_frames_all[mask],
        "x1": G["boxes"][mask][:, 0], "y1": G["boxes"][mask][:, 1],
        "x2": G["boxes"][mask][:, 2], "y2": G["boxes"][mask][:, 3],
        "pid": G["ids"][mask], "cam": G["cams"][mask],
        "height": G["height"][mask],
    })

    frames = sorted(meta["frame"].unique())
    if a.limit:
        frames = frames[:a.limit]
        meta = meta[meta["frame"].isin(frames)].reset_index(drop=True)
        print(f"Smoke test: {len(frames)} frames, {len(meta)} crops")

    hh = meta["height"].to_numpy()
    print(f"\nCrop heights: median {np.median(hh):.0f}px  "
          f"p25 {np.percentile(hh, 25):.0f}  p75 {np.percentile(hh, 75):.0f}")
    print("Proportion of crops each level will actually degrade:")
    for L in LEVELS:
        pct = 100.0 if L == 0 else (hh > L).mean() * 100
        print(f"  {lname(L):<8}{'-' if L == 0 else f'{pct:5.1f}%'}")

    print("\nLoading models ...")
    osnet = load_osnet(models, device)
    irra = IRRAEncoder(a.repo, a.ckpt, device)

    by_frame = {f: g for f, g in meta.groupby("frame")}
    buf = {L: {"os": [], "ir": []} for L in LEVELS}
    feats = {L: {"os": [], "ir": []} for L in LEVELS}
    order = []                                   # row order, must match meta
    t0 = time.time()

    def flush(L):
        b = buf[L]
        if not b["os"]:
            return
        feats[L]["os"].append(encode_osnet(osnet, b["os"], device, a.batch))
        feats[L]["ir"].append(irra.encode_images(b["ir"], max(a.batch // 4, 16)))
        b["os"].clear(); b["ir"].clear()

    print(f"\nProcessing {len(frames)} frames across {len(LEVELS)} levels ...")
    for i, fname in enumerate(frames, 1):
        img = cv2.imread(str(prw / "frames" / fname))
        if img is None:
            continue
        d = by_frame[fname]
        for row in d.itertuples():
            crop = img[row.y1:row.y2, row.x1:row.x2]
            if crop.size == 0:
                continue
            order.append((row.pid, row.cam, row.height))
            for L in LEVELS:
                dc = degrade(crop, L)
                buf[L]["os"].append(prep_osnet(dc))
                buf[L]["ir"].append(irra.preprocess(dc))
        for L in LEVELS:
            if len(buf[L]["os"]) >= a.chunk:
                flush(L)

        if i % 300 == 0 or i == len(frames):
            el = time.time() - t0
            print(f"  {i}/{len(frames)} frames  {len(order)} crops  "
                  f"{i / el:.1f} fps  ETA {(len(frames) - i) / (i / el) / 60:.0f} min")

    for L in LEVELS:
        flush(L)

    g_pid = np.array([o[0] for o in order], np.int32)
    g_cam = np.array([o[1] for o in order], np.int32)
    g_h = np.array([o[2] for o in order], np.int32)
    print(f"\nEncoded {len(g_pid)} crops at each of {len(LEVELS)} levels")

    # Persist the degraded features. Phase 5 evaluates fusion at every level
    # and would otherwise have to repeat this hour of work; saving also allows
    # per-query analysis without recomputation. float16 halves the size at no
    # meaningful cost to cosine similarity.
    if not a.limit:
        outdir = cache / "degraded"
        outdir.mkdir(exist_ok=True)
        total = 0
        for L in LEVELS:
            f = outdir / f"gallery_{lname(L)}.npz"
            np.savez(f,
                     osnet=np.concatenate(feats[L]["os"]).astype(np.float16),
                     irra=np.concatenate(feats[L]["ir"]).astype(np.float16))
            total += f.stat().st_size
        np.savez(outdir / "meta.npz", ids=g_pid, cams=g_cam, height=g_h,
                 levels=np.array(LEVELS))
        print(f"Saved degraded features to {outdir} ({total / 1e6:.0f} MB)")

    # ---- diagnostic: degrade the QUERY as well ------------------------
    # The vision-language image branch improves slightly when the gallery is
    # degraded, which cannot be an information effect. If the improvement is a
    # domain-gap artefact - mild blur moving PRW's sharp crops closer to the
    # softer imagery the model was tuned on - then degrading the query too
    # should remove it.
    print("\nEncoding degraded queries for the diagnostic ...")
    qdir = prw / "query_box"
    q_files = sorted(qdir.glob("*.jpg"))
    q_deg = {L: {"os": [], "ir": []} for L in LEVELS}
    qbuf = {L: {"os": [], "ir": []} for L in LEVELS}

    def qflush(L):
        b = qbuf[L]
        if not b["os"]:
            return
        q_deg[L]["os"].append(encode_osnet(osnet, b["os"], device, a.batch))
        q_deg[L]["ir"].append(irra.encode_images(b["ir"], max(a.batch // 4, 16)))
        b["os"].clear(); b["ir"].clear()

    for p in q_files:
        im = cv2.imread(str(p))
        if im is None:
            continue
        for L in LEVELS:
            dc = degrade(im, L)
            qbuf[L]["os"].append(prep_osnet(dc))
            qbuf[L]["ir"].append(irra.preprocess(dc))
        for L in LEVELS:
            if len(qbuf[L]["os"]) >= a.chunk:
                qflush(L)
    for L in LEVELS:
        qflush(L)
    q_deg = {L: {k: np.concatenate(v) for k, v in d.items()}
             for L, d in q_deg.items()}

    # ------------------------------------------------------ evaluate
    q_os, q_ir, q_pid, q_cam = Q["osnet"], Q["irra"], Q["ids"], Q["cams"]

    frame_to_cam = {str(f): c for f, c in zip(Q["frames"], q_cam)}
    c_pid = C["ids"]
    c_cam = np.array([frame_to_cam.get(str(f), int(str(f).split("s")[0][1:]))
                      for f in C["frames"]], np.int32)
    c_ir = C["irra"]

    rows = []
    print("\n" + "=" * 70)
    print("RESULTS BY RESOLUTION LEVEL")
    print("=" * 70)
    hdr = f"{'level':<9}{'photo(IRRA)':>13}{'photo(OSNet)':>14}{'text':>9}{'degraded':>11}"
    print(hdr)
    print("-" * 70)

    for L in LEVELS:
        g_os = np.concatenate(feats[L]["os"])
        g_ir = np.concatenate(feats[L]["ir"])
        assert len(g_os) == len(g_pid), f"row mismatch at level {L}"

        r_pi = evaluate(q_ir, q_pid, q_cam, g_ir, g_pid, g_cam)
        r_po = evaluate(q_os, q_pid, q_cam, g_os, g_pid, g_cam)
        r_tx = evaluate(c_ir, c_pid, c_cam, g_ir, g_pid, g_cam)

        pct = 100.0 if L == 0 else float((np.array([o[2] for o in order]) > L).mean() * 100)
        print(f"{lname(L):<9}{r_pi['Rank-1']:>12.2f}%{r_po['Rank-1']:>13.2f}%"
              f"{r_tx['Rank-1']:>8.2f}%{'-' if L == 0 else f'{pct:9.1f}%'}")

        # matched degradation: query and gallery reduced to the same level
        r_mi = evaluate(q_deg[L]["ir"], q_pid, q_cam, g_ir, g_pid, g_cam)
        r_mo = evaluate(q_deg[L]["os"], q_pid, q_cam, g_os, g_pid, g_cam)

        for name, r in (("photo_irra", r_pi), ("photo_osnet", r_po),
                        ("text_irra", r_tx),
                        ("photo_irra_matched", r_mi),
                        ("photo_osnet_matched", r_mo)):
            rows.append({"level": lname(L), "target_height": L, "method": name,
                         "pct_degraded": None if L == 0 else round(pct, 1),
                         **{k: round(r[k], 2) for k in
                            ("Rank-1", "Rank-5", "Rank-10", "mAP")},
                         "queries": r["queries"]})

    if not a.limit:
        print("\nRecording per-query outcomes for significance testing ...")
        hits = {}
        for L in LEVELS:
            g_os = np.concatenate(feats[L]["os"]).astype(np.float32)
            g_ir = np.concatenate(feats[L]["ir"]).astype(np.float32)
            hits[f"photo_osnet_{lname(L)}"] = evaluate_hits(
                q_os, q_pid, q_cam, g_os, g_pid, g_cam)
            hits[f"photo_irra_{lname(L)}"] = evaluate_hits(
                q_ir, q_pid, q_cam, g_ir, g_pid, g_cam)
            hits[f"text_irra_{lname(L)}"] = evaluate_hits(
                c_ir, c_pid, c_cam, g_ir, g_pid, g_cam)
        np.savez_compressed(cache / "phase4_hits.npz", **hits)
        print(f"Saved {cache / 'phase4_hits.npz'}")

    df = pd.DataFrame(rows)
    out = cache / f"phase4_resolution{'_smoke' if a.limit else ''}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")

    # ------------------------------------------------------ crossing point
    piv = df.pivot(index="target_height", columns="method", values="Rank-1")
    piv = piv.reindex(LEVELS)
    if "photo_irra_matched" in piv.columns:
        print("\n" + "=" * 70)
        print("DIAGNOSTIC: query degraded to the same level as the gallery")
        print("=" * 70)
        print(f"{'level':<9}{'gallery only':>14}{'both degraded':>16}{'difference':>13}")
        for L in LEVELS:
            d = piv["photo_irra_matched"][L] - piv["photo_irra"][L]
            print(f"{lname(L):<9}{piv['photo_irra'][L]:>13.2f}%"
                  f"{piv['photo_irra_matched'][L]:>15.2f}%{d:>12.2f} pp")
        bump = piv["photo_irra"][96] - piv["photo_irra"][0]
        bump_m = piv["photo_irra_matched"][96] - piv["photo_irra_matched"][0]
        print(f"\nGain from degrading to 96px:")
        print(f"  gallery only   {bump:+.2f} pp")
        print(f"  both degraded  {bump_m:+.2f} pp")
        if bump > 0.5 and bump_m <= bump / 2:
            print("  -> the gain largely disappears when both sides are degraded,")
            print("     consistent with a domain-gap artefact rather than an")
            print("     information effect.")
        elif bump > 0.5:
            print("  -> the gain persists, so a domain-gap explanation is not")
            print("     supported; investigate further before writing it up.")
    print("\n" + "=" * 70)
    print("RANK-1 BY LEVEL")
    print("=" * 70)
    print(piv.to_string())

    print("\n" + "=" * 70)
    print("READING THE RESULT")
    print("=" * 70)
    for img_col, label in (("photo_irra", "vision-language"),
                           ("photo_osnet", "appearance")):
        gap = piv[img_col] - piv["text_irra"]
        print(f"\n{label} photograph minus description, by level:")
        for L in LEVELS:
            print(f"  {lname(L):<8}{gap[L]:+7.2f} pp")
        below = [L for L in LEVELS if L != 0 and gap[L] < 0]
        if below:
            print(f"  -> description overtakes the photograph at "
                  f"{lname(max(below))} and below")
        elif gap[LEVELS[-1]] < gap[0]:
            print(f"  -> the gap narrows from {gap[0]:+.1f} to "
                  f"{gap[LEVELS[-1]]:+.1f} pp but does not cross")
        else:
            print("  -> no narrowing observed; check the degradation is applied")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = list(range(len(LEVELS)))
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for col, lab, st in (("photo_irra", "Photograph (vision-language)", "-o"),
                             ("photo_osnet", "Photograph (appearance)", "-s"),
                             ("text_irra", "Description", "-^")):
            ax.plot(x, [piv[col][L] for L in LEVELS], st, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([lname(L) for L in LEVELS])
        ax.set_xlabel("gallery crop height after degradation")
        ax.set_ylabel("Rank-1 (%)")
        ax.set_title("Retrieval accuracy against gallery resolution")
        ax.grid(alpha=.3); ax.legend()
        fig.tight_layout()
        png = cache / f"phase4_resolution{'_smoke' if a.limit else ''}.png"
        fig.savefig(png, dpi=150)
        print(f"\nSaved {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
