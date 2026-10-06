"""
Phase 3b - Text branch: IRRA encoder.

IRRA (Jiang & Ye, CVPR 2023) is CLIP ViT-B/16 fine-tuned for text-to-image
person retrieval. Published on CUHK-PEDES at Rank-1 73.38 against 68.19 for
the same CLIP backbone untuned, so the fine-tuning is worth the trouble.

This module loads the released checkpoint WITHOUT running IRRA's own loader,
for two reasons:

  1. IRRA's build_CLIP_from_openai_pretrained() downloads OpenAI's CLIP
     archive and opens it with torch.jit.load. That archive is from 2021 and
     the JIT format has moved on.
  2. It then calls torch.load() with no weights_only argument. Since PyTorch
     2.6 that defaults to True and refuses these checkpoints outright.

Neither is necessary: the IRRA checkpoint already contains every CLIP weight
under the base_model. prefix, so the architecture can be built from the
checkpoint itself and the network is never touched.

Verified against the released checkpoint:
  img_size (384, 128), stride 16 -> 24x8 = 192 patches + 1 cls = 193
  base_model.visual.positional_embedding is (193, 768)   <- matches
  367 tensors, of which mlm_head.* and cross_modal.* are training-only

Self-test:
  python phase3_irra.py --repo .\\IRRA --ckpt .\\models\\irra\\best.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

IMG_SIZE = (384, 128)          # height, width - from configs.yaml
STRIDE = 16
CONTEXT_LEN = 77

# CLIP's own normalisation, not ImageNet's
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def _add_repo(repo: Path):
    repo = repo.resolve()
    if not (repo / "model" / "clip_model.py").exists():
        sys.exit(f"ERROR: {repo} does not look like the IRRA repo.\n"
                 f"  git clone https://github.com/anosorae/IRRA.git")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def load_state(ckpt_path: Path):
    """Return the base_model.* tensors with the prefix stripped."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    base = {k[len("base_model."):]: v
            for k, v in sd.items() if k.startswith("base_model.")}
    if not base:
        sys.exit("ERROR: no base_model.* tensors in the checkpoint. "
                 "Is this the IRRA release?")
    return base


def build_from_state(base_sd, repo: Path, device):
    """
    Derive the architecture from the checkpoint's own tensor shapes, then
    instantiate IRRA's CLIP class directly. No download, no jit.
    """
    from model.clip_model import CLIP

    cfg = dict(
        embed_dim=base_sd["text_projection"].shape[1],
        image_resolution=IMG_SIZE,
        vision_layers=len([k for k in base_sd
                           if k.startswith("visual.")
                           and k.endswith(".attn.in_proj_weight")]),
        vision_width=base_sd["visual.conv1.weight"].shape[0],
        vision_patch_size=base_sd["visual.conv1.weight"].shape[-1],
        stride_size=STRIDE,
        context_length=base_sd["positional_embedding"].shape[0],
        vocab_size=base_sd["token_embedding.weight"].shape[0],
        transformer_width=base_sd["ln_final.weight"].shape[0],
        transformer_heads=base_sd["ln_final.weight"].shape[0] // 64,
        transformer_layers=len({k.split(".")[2] for k in base_sd
                                if k.startswith("transformer.resblocks")}),
    )
    print("  architecture derived from checkpoint:")
    for k in ("embed_dim", "vision_layers", "vision_width",
              "vision_patch_size", "transformer_layers"):
        print(f"    {k:<20}{cfg[k]}")

    model = CLIP(**cfg)
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    missing = [m for m in missing if "position_ids" not in m]
    if missing:
        print(f"  WARNING: {len(missing)} tensors missing, e.g. {missing[:3]}")
    if unexpected:
        print(f"  note: {len(unexpected)} unused tensors, e.g. {unexpected[:2]}")
    if not missing:
        print(f"  all {len(base_sd)} CLIP tensors loaded")

    return model.to(device).eval(), cfg


class IRRAEncoder:
    """Encodes person crops and captions into one 512-d space."""

    def __init__(self, repo, ckpt, device=None):
        self.repo = _add_repo(Path(repo))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading IRRA checkpoint ({Path(ckpt).stat().st_size / 1e9:.2f} GB) ...")
        base_sd = load_state(Path(ckpt))
        self.model, self.cfg = build_from_state(base_sd, self.repo, self.device)

        from utils.simple_tokenizer import SimpleTokenizer
        self.tokenizer = SimpleTokenizer()
        self.sot = self.tokenizer.encoder["<|startoftext|>"]
        self.eot = self.tokenizer.encoder["<|endoftext|>"]
        print(f"  device: {self.device}")

    # ---------------------------------------------------------- text
    def tokenize(self, captions):
        out = torch.zeros(len(captions), CONTEXT_LEN, dtype=torch.long)
        for i, c in enumerate(captions):
            toks = [self.sot] + self.tokenizer.encode(c) + [self.eot]
            if len(toks) > CONTEXT_LEN:                 # truncate, keep <eot>
                toks = toks[:CONTEXT_LEN]
                toks[-1] = self.eot
            out[i, :len(toks)] = torch.tensor(toks)
        return out

    @torch.no_grad()
    def encode_text(self, captions, batch=256):
        feats = []
        for i in range(0, len(captions), batch):
            ids = self.tokenize(captions[i:i + batch]).to(self.device)
            x = self.model.encode_text(ids)
            # IRRA takes the feature at the <eot> position
            x = x[torch.arange(x.shape[0]), ids.argmax(dim=-1)].float()
            feats.append(torch.nn.functional.normalize(x, dim=1).cpu().numpy())
        return np.concatenate(feats).astype(np.float32)

    # ---------------------------------------------------------- image
    @staticmethod
    def preprocess(bgr):
        import cv2
        im = cv2.resize(bgr, (IMG_SIZE[1], IMG_SIZE[0]),
                        interpolation=cv2.INTER_CUBIC)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        im = (im - CLIP_MEAN) / CLIP_STD
        return im.transpose(2, 0, 1)

    @torch.no_grad()
    def encode_images(self, arrays, batch=64):
        """arrays: list of preprocessed CHW float32 arrays."""
        feats = []
        for i in range(0, len(arrays), batch):
            x = torch.from_numpy(np.stack(arrays[i:i + batch])).to(self.device)
            f = self.model.encode_image(x)
            f = f[:, 0, :].float()                      # class token
            feats.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        return np.concatenate(feats).astype(np.float32)


def self_test(args):
    enc = IRRAEncoder(args.repo, args.ckpt)

    print("\n--- text encoder ---")
    caps = [
        "A woman wearing a red jacket and blue jeans.",
        "A man in a black suit carrying a briefcase.",
        "A person in a bright red coat with dark denim trousers.",
    ]
    tf = enc.encode_text(caps)
    print(f"  shape {tf.shape}  norm {np.linalg.norm(tf[0]):.3f} (expect 1.0)")
    sim = tf @ tf.T
    print(f"  sim(red jacket/blue jeans, red coat/dark denim) = {sim[0, 2]:.3f}")
    print(f"  sim(red jacket/blue jeans, black suit)          = {sim[0, 1]:.3f}")
    ok_text = sim[0, 2] > sim[0, 1]
    print(f"  paraphrase scores higher than unrelated: "
          f"{'YES' if ok_text else 'NO'}")

    print("\n--- image encoder ---")
    rng = np.random.default_rng(0)
    fake = [((rng.random((IMG_SIZE[0], IMG_SIZE[1], 3)) * 255).astype(np.uint8))
            for _ in range(4)]
    arrs = [enc.preprocess(f) for f in fake]
    imf = enc.encode_images(arrs)
    print(f"  shape {imf.shape}  norm {np.linalg.norm(imf[0]):.3f} (expect 1.0)")

    print("\n--- shared space ---")
    cross = imf @ tf.T
    print(f"  image-text similarity matrix {cross.shape}, "
          f"range [{cross.min():.3f}, {cross.max():.3f}]")

    print("\n" + "=" * 58)
    if imf.shape[1] == tf.shape[1] == 512 and ok_text:
        print("SELF-TEST PASSED")
        print("Both encoders produce 512-d unit vectors in one space.")
        print("Proceed to encoding the PRW gallery.")
    else:
        print("SELF-TEST FAILED")
        if imf.shape[1] != tf.shape[1]:
            print("  image and text dimensions differ - check embed_dim")
        if not ok_text:
            print("  text embeddings are not semantically ordered;")
            print("  the checkpoint may not have loaded correctly")
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="IRRA", help="path to the cloned IRRA repo")
    ap.add_argument("--ckpt", default="models/irra/best.pth")
    self_test(ap.parse_args())
