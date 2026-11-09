"""A human dataset of the rows at or above an ascension, from one or more built datasets
(`human_dataset.py`'s <out>, with train/ and val/), without encoding the runs again.

    .venv/bin/python python/human/subset_dataset.py ~/sts-data/human-defect ~/sts-data/human-defect-events \
        --min-asc 15 --out ~/sts-data/human-defect-a15

Reads each dataset's train/ and val/ rows in order (one sequential pass), keeps those whose
ascension is at least --min-asc (a val row stays held out), writes them to <out> unshuffled and
then shuffles them into <out>/train and <out>/val as `human_dataset.shuffle` does. The kinds keep
their numbers (`human_dataset.KINDS`), so choices and events can be merged.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import json

import numpy as np

from human.human_dataset import KINDS, ROW_FILES, SMALL, shuffle


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="+")
    p.add_argument("--min-asc", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chunk", type=int, default=100_000)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = {k: open(os.path.join(a.out, f"{k}.bin"), "wb") for k, _, _ in ROW_FILES}
    small = {k: [] for k in (*SMALL, "val")}
    meta = None
    for ds in a.datasets:
        for part, is_val in (("train", False), ("val", True)):
            d = os.path.join(ds, part)
            with open(os.path.join(d, "meta.json")) as f:
                m = json.load(f)
            if meta is None:
                meta = m
            assert all(m[k] == meta[k] for k in ("ints", "floats", "mask_bytes")), f"{d}: other row lengths"
            n = m["n"]
            src = {k: np.memmap(os.path.join(d, f"{k}.bin"), dtype=dt, mode="r", shape=(n, m[w])) for k, dt, w in ROW_FILES}
            sm = {k: np.load(os.path.join(d, f"{k}.npy")) for k in SMALL if os.path.exists(os.path.join(d, f"{k}.npy"))}
            keep = sm["asc"] >= a.min_asc
            for start in range(0, n, a.chunk):
                sel = keep[start:start + a.chunk]
                if sel.any():
                    for k in src:
                        np.asarray(src[k][start:start + a.chunk])[sel].tofile(files[k])
            for k in sm:  # (datasets from before 2026-10-27 have no outcome arrays)
                small[k].append(sm[k][keep])
            small["val"].append(np.full(int(keep.sum()), is_val))
            print(f"{d}: {int(keep.sum()):,} of {n:,} rows at A{a.min_asc}+", flush=True)
    for f in files.values():
        f.close()
    for k, v in small.items():
        if v:
            np.save(os.path.join(a.out, f"{k}.npy"), np.concatenate(v))
    kind = np.concatenate(small["kind"])
    n = len(kind)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({**{k: meta[k] for k in ("ints", "floats", "actions", "mask_bytes")}, "n": n, "won_only": True, "kinds": KINDS,
                   "min_asc": a.min_asc, "from": a.datasets}, f, indent=1)
    print(f"{n:,} rows; by kind: " + ", ".join(f"{k} {int((kind == v).sum()):,}" for k, v in KINDS.items()), flush=True)
    shuffle(a.out)


if __name__ == "__main__":
    main()
