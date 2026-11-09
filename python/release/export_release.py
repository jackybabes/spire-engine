"""Writes a run checkpoint's network as a release weights file, and reads one back.

    python python/release/export_release.py python/runs/ironclad-runs-habits-12/update_200.pt ironclad-v1.safetensors

The file is safetensors, written by hand (no extra library): an 8-byte little-endian header
length, a JSON header naming each tensor's dtype, shape and byte range, then the raw
little-endian float32 tensors. Only the network's weights go in (no optimizer state), and the
header's `__metadata__` (strings only) carries the architecture (d, layers, heads), the encoder's
signature (RUN_INT_LEN, RUN_FLOAT_LEN, ACTIONS, ENCODING_VERSION), the source checkpoint's
update and steps, the git commit and the character.

`load_release(path, device)` builds the network from such a file and refuses one whose encoder
signature doesn't match this build's `sts` module: a network trained on other inputs would read
the wrong numbers without any error.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import json
import os
import struct
import subprocess

import numpy as np
import torch

from core.run_model import RunNet
from core.sts_env import ACTIONS, ENCODING_VERSION, RUN_FLOAT_LEN, RUN_INT_LEN

FORMAT = "sts-run-net"


def signature():
    """This build's encoder signature, as the weights file stores it."""
    return {"run_int_len": str(RUN_INT_LEN), "run_float_len": str(RUN_FLOAT_LEN),
            "actions": str(ACTIONS), "encoding_version": str(ENCODING_VERSION)}


def git_commit():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("-dirty" if dirty else "")


def write_safetensors(path, tensors, metadata):
    """tensors: name -> float32 tensor; metadata: str -> str."""
    header, blobs, at = {"__metadata__": metadata}, [], 0
    for name in sorted(tensors):
        a = tensors[name].detach().cpu().to(torch.float32).contiguous().numpy().astype("<f4", copy=False)
        b = a.tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [at, at + len(b)]}
        blobs.append(b)
        at += len(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)  # the spec allows trailing spaces; keeps the data 8-byte aligned
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def read_safetensors(path):
    """-> (name -> float32 tensor, metadata)."""
    with open(path, "rb") as f:
        data = f.read()
    (n,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8 + n])
    metadata = header.pop("__metadata__", {})
    base, tensors = 8 + n, {}
    for name, t in header.items():
        if t["dtype"] != "F32":
            raise ValueError(f"{path}: {name} is {t['dtype']}, expected F32")
        lo, hi = t["data_offsets"]
        a = np.frombuffer(data, dtype="<f4", count=(hi - lo) // 4, offset=base + lo).reshape(t["shape"])
        tensors[name] = torch.from_numpy(a.astype(np.float32))
    return tensors, metadata


def export(net, ck, path, character="ironclad"):
    """Writes `net` (built from checkpoint `ck`, see train_runs.load) to `path`."""
    a = ck["args"]
    if getattr(net, "unblended", False):
        raise ValueError("the browser's forward pass (crates/sts-net) has no unblended pieces yet")
    # (the human-outcome and next-move heads train the network and play no part in it: left out)
    state = {k: v for k, v in net.state_dict().items() if not k.startswith(("outcome.", "move_head.", "fight_outcome_head."))}
    # The lookup's inputs (train_runs.py --lookup): the browser's forward pass has no lookup yet, so only a network that
    # never learned to use it (its weights still zero) can go out, without them.
    lookup = {k: v for k, v in state.items() if k.startswith(("lookup_opt.", "lookup_run.", "lookup_pick."))}
    if any(v.abs().sum() > 0 for v in lookup.values()):
        raise ValueError("the network uses the deck lookup (train_runs.py --lookup), which crates/sts-net doesn't compute yet")
    state = {k: v for k, v in state.items() if k not in lookup}
    metadata = {"format": FORMAT, "d": str(a.get("d", 128)), "layers": str(a.get("layers", 3)),
                "heads": str(a.get("heads", 4)), **signature(),
                "source": str(a.get("name", "")), "update": str(ck.get("update", "")),
                "steps": str(ck.get("steps", "")), "git_commit": git_commit(), "character": character,
                "parameters": str(sum(v.numel() for v in state.values()))}
    write_safetensors(path, state, metadata)
    return metadata


def load_release(path, device="cpu"):
    """-> (RunNet in eval mode, metadata). Raises ValueError on another encoder's file."""
    tensors, meta = read_safetensors(path)
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: not a run network release file (format {meta.get('format')!r})")
    want = signature()
    wrong = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if wrong:
        raise ValueError(f"{path} was made for another encoding: " +
                         ", ".join(f"{k} {got} (this build {v})" for k, (got, v) in wrong.items()))
    net = RunNet(d=int(meta["d"]), layers=int(meta["layers"]), heads=int(meta["heads"]))
    # (A release leaves out the deck lookup's inputs, all zero: `export`.)
    for k, v in net.state_dict().items():
        if k.startswith(("lookup_opt.", "lookup_run.", "lookup_pick.")):
            tensors.setdefault(k, torch.zeros_like(v))
    net.load_state_dict(tensors, strict=True)
    return net.to(device).eval(), meta


def from_checkpoint(path, device="cpu"):
    """-> (RunNet with the checkpoint's weights, checkpoint), as `train_runs.py --eval` loads it:
    `train_runs.load` only builds the network, and loading the state grows the embedding tables
    to this build's vocabulary (the other characters' ids; rows a network never trained on)."""
    from training.train_runs import load
    net, ck = load(path, device)
    net.load_state_dict(ck["net"])
    return net.eval(), ck


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("out")
    p.add_argument("--character", default="ironclad", choices=["ironclad", "silent", "defect", "watcher"])
    args = p.parse_args()
    net, ck = from_checkpoint(args.checkpoint)
    meta = export(net, ck, args.out, args.character)
    load_release(args.out)
    print(f"wrote {args.out} ({os.path.getsize(args.out):,} bytes): {json.dumps(meta)}")


if __name__ == "__main__":
    main()
