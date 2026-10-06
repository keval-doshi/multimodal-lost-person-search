"""
Phase 6b - Search interface.

A simple screen: upload a photograph, type a clothing description, and see
the ranked sightings and which cameras saw the person.

  python phase6_app.py --root .\\dataset

Then open the address it prints, normally http://127.0.0.1:7860

This demonstrates the pipeline; it is not a contribution in itself and is
deliberately kept small. The configuration is the one Phase 5 selected:
appearance branch, alpha = 0.45.

Either input may be left empty. With only a photograph it runs image
retrieval, with only a description it runs text retrieval, and with both it
fuses them. That makes the complementarity visible to the user directly.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

ALPHA = 0.45


def zrow(v):
    return (v - v.mean()) / (v.std() + 1e-8)


def main():
    import gradio as gr
    import cv2
    import torch
    from phase3_encode import load_osnet, prep_osnet, encode_osnet
    from phase3_irra import IRRAEncoder
    from phase6_timeline import build_timeline, parse_frame
    from phase3_baseline import load_split

    P = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_argument("--root", default="dataset")
    P.add_argument("--repo", default="IRRA")
    P.add_argument("--ckpt", default="models/irra/best.pth")
    P.add_argument("--share", action="store_true")
    a_ = P.parse_args()

    root = Path(a_.root).resolve()
    prw = root / "PRW"
    cache = root.parent / "cache"
    models = root.parent / "models"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    G = np.load(cache / "gallery.npz", allow_pickle=True)
    sp = load_split(prw, "frame_test.mat")
    mask = np.array([f in sp for f in G["frames"]]) if sp else np.ones(len(G["frames"]), bool)
    g_os = G["osnet"][mask].astype(np.float32)
    g_ir = G["irra"][mask].astype(np.float32)
    g_pid, g_cam = G["ids"][mask], G["cams"][mask]
    g_frames, g_boxes = G["frames"][mask], G["boxes"][mask]
    print(f"Gallery: {len(g_pid)} crops, {len(set(g_cam.tolist()))} cameras")

    osnet = load_osnet(models, device)
    irra = IRRAEncoder(a_.repo, a_.ckpt, device)

    def crop_of(i):
        img = cv2.imread(str(prw / "frames" / str(g_frames[i])))
        if img is None:
            return None
        x1, y1, x2, y2 = g_boxes[i]
        c = img[int(y1):int(y2), int(x1):int(x2)]
        return cv2.cvtColor(c, cv2.COLOR_BGR2RGB) if c.size else None

    def search(photo, description, alpha, top_k):
        if photo is None and not (description or "").strip():
            return [], "Provide a photograph, a description, or both.", ""

        parts, used = [], []
        if photo is not None:
            bgr = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)
            f = encode_osnet(osnet, [prep_osnet(bgr)], device, 1)
            parts.append((alpha, zrow(f[0] @ g_os.T)))
            used.append("photograph")
        if (description or "").strip():
            t = irra.encode_text([description.strip()])
            parts.append((1 - alpha, zrow(t[0] @ g_ir.T)))
            used.append("description")

        if len(parts) == 1:                    # single source, weight is moot
            scores = parts[0][1]
        else:
            scores = sum(w * s for w, s in parts)

        st = build_timeline(scores, g_frames, g_cam, g_pid,
                            q_cam=-1, top_k=int(top_k))

        gallery, lines = [], []
        by_cam = {}
        for rank, s in enumerate(st, 1):
            im = crop_of(s["gallery_index"])
            if im is not None:
                gallery.append((im, f"#{rank}  c{s['camera']}  "
                                    f"score {s['score']:.2f}"))
            by_cam.setdefault(s["camera"], []).append(s)
            lines.append(f"| {rank} | c{s['camera']} | {s['segment']} | "
                         f"{s['frame']} | {s['score']:.3f} |")

        table = ("| rank | camera | segment | frame | score |\n"
                 "|---|---|---|---|---|\n" + "\n".join(lines))

        summary = [f"**Searched using: {' and '.join(used)}**", ""]
        if len(parts) > 1:
            summary.append(f"Balance: {alpha:.2f} photograph, "
                           f"{1 - alpha:.2f} description")
            summary.append("")
        summary.append(f"**{len(st)} candidate sightings across "
                       f"{len(by_cam)} camera(s)**")
        summary.append("")
        for c in sorted(by_cam):
            segs = sorted({x['segment'] for x in by_cam[c]})
            best = max(x['score'] for x in by_cam[c])
            summary.append(f"- **Camera {c}** — {len(by_cam[c])} sighting(s), "
                           f"segment(s) {', '.join(map(str, segs))}, "
                           f"best score {best:.2f}")
        summary.append("")
        summary.append("*Frame numbers restart in each segment and the cameras "
                       "share no clock, so sightings are ordered within a "
                       "segment but not across cameras.*")
        return gallery, "\n".join(summary), table

    with gr.Blocks(title="Multimodal Lost Person Search") as demo:
        gr.Markdown("# Multimodal Lost Person Search")
        gr.Markdown(
            "Upload a photograph of the missing person, describe what they "
            "were wearing, or do both. The system searches recorded footage "
            "from six cameras and reports where the person may have been seen."
            "\n\nWith the appearance model, combining the two raised mean "
            "average precision by about 10 points over the photograph alone "
            "on this benchmark."
        )

        with gr.Row():
            with gr.Column(scale=1):
                photo = gr.Image(label="Photograph of the person", type="numpy")
                desc = gr.Textbox(
                    label="What were they wearing?",
                    placeholder="A woman in a white shirt and grey trousers, "
                                "black hair, carrying a dark bag.",
                    lines=3)
                alpha = gr.Slider(0, 1, ALPHA, step=0.05,
                                  label="Balance  (0 = description only, "
                                        "1 = photograph only)")
                top_k = gr.Slider(5, 40, 20, step=5, label="Sightings to return")
                go = gr.Button("Search", variant="primary")
                gr.Markdown(
                    f"The default balance of {ALPHA} was tuned on validation "
                    "identities. There is no face-recognition component: "
                    "matching relies mainly on clothing, build and appearance.")
            with gr.Column(scale=2):
                summary = gr.Markdown()
                results = gr.Gallery(label="Candidate sightings", columns=5,
                                     height=420)
                table = gr.Markdown()

        go.click(search, [photo, desc, alpha, top_k],
                 [results, summary, table])

    demo.launch(share=a_.share)


if __name__ == "__main__":
    main()
