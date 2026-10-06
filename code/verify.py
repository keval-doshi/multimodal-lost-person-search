"""
Session 1 dataset verification.

Checks that PRW, PRW-TBPS and Market-1501 are present, complete, and that the
PRW-TBPS captions actually join to the PRW image files. That join is the single
point of failure for the multimodal benchmark, so it is tested explicitly.

Usage:
    python verify_datasets.py --root datasets

Expected layout under --root:
    PRW/frames/  PRW/annotations/  PRW/query_box/
    PRW-TBPS/PRW-TBPS_train  PRW-TBPS/PRW-TBPS_test
    Market-1501/bounding_box_train/  bounding_box_test/  query/
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Published reference counts. Sources:
#   PRW           - Zheng et al., CVPR 2017 (arXiv:1604.02531)
#   PRW-TBPS      - Zhang et al., 2021 (arXiv:2109.12965)
#   Market-1501   - Zheng et al., ICCV 2015
EXPECTED = {
    "prw_frames": 11816,
    "prw_queries": 2057,
    "prw_ids": 932,
    "tbps_test_entries": 2057,
    "tbps_test_ids": 450,
    "tbps_train_boxes": 14897,
    "tbps_train_ids": 483,
    "market_train": 12936,
    "market_test": 19732,
    "market_query": 3368,
}

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results = []


def check(label, got, want=None, tolerance=0):
    """Record one check. If `want` is None the value is informational only."""
    if want is None:
        results.append((PASS, label, f"{got}"))
        return True
    ok = abs(got - want) <= tolerance
    status = PASS if ok else FAIL
    detail = f"{got} (expected {want})" if not ok else f"{got}"
    results.append((status, label, detail))
    return ok


def note(status, label, detail):
    results.append((status, label, detail))


# --------------------------------------------------------------------------
# PRW
# --------------------------------------------------------------------------
def verify_prw(root: Path):
    prw = root / "PRW"
    if not prw.is_dir():
        note(FAIL, "PRW", f"directory not found at {prw}")
        return None

    frames = sorted((prw / "frames").glob("*.jpg"))
    check("PRW frames", len(frames), EXPECTED["prw_frames"])

    anns = sorted((prw / "annotations").glob("*.mat"))
    check("PRW annotation files", len(anns), EXPECTED["prw_frames"])

    qbox = sorted((prw / "query_box").glob("*.jpg"))
    check("PRW query boxes", len(qbox), EXPECTED["prw_queries"])

    # Camera distribution across frames, e.g. c1s3_016471.jpg -> c1
    cams = Counter(m.group(1) for f in frames
                   if (m := re.match(r"(c\d+)s", f.name)))
    if cams:
        note(PASS, "PRW cameras", ", ".join(f"{k}:{v}" for k, v in sorted(cams.items())))
        if len(cams) != 6:
            note(WARN, "PRW camera count", f"{len(cams)} distinct cameras (expected 6)")

    return {f.name for f in frames}


# --------------------------------------------------------------------------
# PRW-TBPS
# --------------------------------------------------------------------------
def load_tbps(path: Path):
    """Load a PRW-TBPS annotation file. Note the key is misspelled upstream."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path.name}: expected a JSON list, got {type(data).__name__}")
    return data


def caption_key(entry):
    """Upstream releases spell this 'desrciption'. Accept either spelling."""
    for k in ("desrciption", "description"):
        if k in entry:
            return k
    raise KeyError(f"no caption field in entry: {list(entry.keys())}")


def find_tbps(root: Path, split: str):
    """The annotation files may sit in PRW-TBPS/, in PRW/, or at the root."""
    name = f"PRW-TBPS_{split}"
    for candidate in (root / "PRW-TBPS" / name, root / "PRW" / name, root / name):
        if candidate.exists():
            return candidate
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def verify_tbps(root: Path, frame_names):
    for split, n_key, id_key in (("test", "tbps_test_entries", "tbps_test_ids"),
                                 ("train", "tbps_train_boxes", "tbps_train_ids")):
        path = find_tbps(root, split)
        if path is None:
            status = FAIL if split == "test" else WARN
            note(status, f"PRW-TBPS {split}", "file missing")
            continue

        try:
            data = load_tbps(path)
        except (ValueError, json.JSONDecodeError) as exc:
            note(FAIL, f"PRW-TBPS {split}", str(exc))
            continue

        key = caption_key(data[0])
        if key == "description":
            note(WARN, f"PRW-TBPS {split} key",
                 "field is 'description'; upstream uses 'desrciption' - check your loader")

        check(f"PRW-TBPS {split} entries", len(data), EXPECTED[n_key])
        check(f"PRW-TBPS {split} identities",
              len({e["id"] for e in data}), EXPECTED[id_key])

        caps = [c for e in data for c in e[key]]
        per_entry = Counter(len(e[key]) for e in data)
        note(PASS, f"PRW-TBPS {split} captions",
             f"{len(caps)} total, {dict(per_entry)} per entry")

        empty = sum(1 for c in caps if not c.strip())
        if empty:
            note(WARN, f"PRW-TBPS {split} empty captions", str(empty))

        # THE CRITICAL CHECK: do these captions point at real PRW frames?
        if frame_names is not None:
            missing = [e["pic_path"] for e in data if e["pic_path"] not in frame_names]
            if missing:
                note(FAIL, f"PRW-TBPS {split} -> PRW join",
                     f"{len(missing)} of {len(data)} captions reference missing frames "
                     f"(e.g. {missing[0]})")
            else:
                note(PASS, f"PRW-TBPS {split} -> PRW join",
                     f"all {len(data)} captions resolve to real frames")


# --------------------------------------------------------------------------
# Market-1501
# --------------------------------------------------------------------------
def verify_market(root: Path):
    mk = root / "Market-1501"
    if not mk.is_dir():
        mk = next((p for p in root.glob("Market-1501*") if p.is_dir()), None)
        if mk is None:
            note(WARN, "Market-1501", "directory not found")
            return

    # The official zip unpacks to a nested Market-1501-v15.09.15/ folder.
    if not (mk / "query").is_dir():
        nested = next((p for p in mk.glob("Market-1501*") if (p / "query").is_dir()), None)
        if nested is not None:
            note(PASS, "Market-1501 root", f"using nested {nested.name}")
            mk = nested

    for sub, key in (("bounding_box_train", "market_train"),
                     ("bounding_box_test", "market_test"),
                     ("query", "market_query")):
        d = mk / sub
        if not d.is_dir():
            note(FAIL, f"Market-1501/{sub}", "missing")
            continue
        check(f"Market-1501 {sub}", len(list(d.glob("*.jpg"))), EXPECTED[key])


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="datasets", type=Path,
                    help="directory containing PRW/, PRW-TBPS/, Market-1501/")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    print(f"Verifying datasets under: {root}\n")

    frame_names = verify_prw(root)
    verify_tbps(root, frame_names)
    verify_market(root)

    width = max(len(label) for _, label, _ in results) + 2
    for status, label, detail in results:
        print(f"[{status}] {label:<{width}} {detail}")

    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    print(f"\n{len(results)} checks - {n_fail} failed, {n_warn} warnings")

    if n_fail:
        print("\nFix the failures before Session 2. A broken caption-to-frame join "
              "means the multimodal benchmark cannot be built.")
    else:
        print("\nDatasets verified. Ready for detection (Session 2).")

    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
