"""Checks for export_release.py (plain asserts, no pytest needed):

    python python/tests/test_export.py                                   # a small random network
    python python/tests/test_export.py python/runs/ironclad-runs-habits-12/update_200.pt    # and a real checkpoint

The exported network's logits and values must equal the checkpoint's exactly on a batch of run
observations (fights and the choices between them), and a file made for another encoding must
be refused."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import os
import sys
import tempfile

import numpy as np
import torch

import release.export_release as export_release
from release.export_release import export, from_checkpoint, load_release, read_safetensors, write_safetensors
from core.run_model import COMBAT, RUN_INT_AT, RunNet
from core.sts_env import RunEnv


def observations(n=64, steps=400, seed=7):
    """Rows from runs played by random legal moves, taken every few steps: both screens."""
    env, rng = RunEnv(n, seed=seed), np.random.default_rng(seed)
    ints, floats = [], []
    for k in range(steps):
        obs = env.observe()
        if k % 50 == 49:
            ints.append(obs["ints"])
            floats.append(obs["floats"])
        env.step([rng.choice(np.flatnonzero(m)) for m in obs["mask"]])
    return torch.from_numpy(np.concatenate(ints)).long(), torch.from_numpy(np.concatenate(floats))


def outputs(net, ints, floats):
    net.eval()
    with torch.no_grad():
        return net(ints, floats)


def check(net, ck, ints, floats, tmp):
    path = os.path.join(tmp, "net.safetensors")
    meta = export(net, ck, path)
    got, got_meta = load_release(path)
    assert got_meta == meta
    (l0, v0), (l1, v1) = outputs(net, ints, floats), outputs(got, ints, floats)
    assert torch.equal(l0, l1) and torch.equal(v0, v1), (l0 - l1).abs().max()
    # The file holds only the weights, as float32.
    n = sum(p.numel() for p in net.state_dict().values())
    assert os.path.getsize(path) < 4 * n + 64 * 1024, os.path.getsize(path)
    return path


ints, floats = observations()
combat = (ints[:, RUN_INT_AT["screen"][0]] == COMBAT).sum().item()
assert 0 < combat < len(ints), (combat, len(ints))

with tempfile.TemporaryDirectory() as tmp:
    torch.manual_seed(0)
    net = RunNet(d=32, layers=2, heads=2)
    with torch.no_grad():  # some output layers start at zero; give every weight a value
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        for lin in (net.lookup_opt, net.lookup_run):  # (but the deck lookup's: the browser has none yet, and export refuses it)
            lin.weight.zero_()
            lin.bias.zero_()
        net.lookup_pick.weight.zero_()
    ck = {"args": {"name": "test", "d": 32, "layers": 2, "heads": 2}, "update": 3, "steps": 1000}
    path = check(net, ck, ints, floats, tmp)

    # Another encoding is refused: each signature field in turn.
    tensors, meta = read_safetensors(path)
    for key in export_release.signature():
        bad = dict(meta, **{key: str(int(meta[key]) + 1)})
        write_safetensors(path + ".bad", tensors, bad)
        try:
            load_release(path + ".bad")
            raise AssertionError(f"loaded a file with another {key}")
        except ValueError as e:
            assert key in str(e), e
    # A missing tensor is refused too (strict load).
    write_safetensors(path + ".bad", {k: v for k, v in tensors.items() if k != "run_pick.weight"}, meta)
    try:
        load_release(path + ".bad")
        raise AssertionError("loaded a file with a tensor missing")
    except RuntimeError:
        pass

    for ckpt in sys.argv[1:]:
        net, ck = from_checkpoint(ckpt)
        path = check(net, ck, ints, floats, tmp)
        # The file holds the checkpoint's own weights (embedding tables grown with extra rows).
        tensors, _ = read_safetensors(path)
        for k, v in ck["net"].items():
            v = v.float().cpu()
            assert torch.equal(tensors[k][tuple(slice(0, n) for n in v.shape)], v), k
        print(f"{ckpt}: exported network matches")

print("export_release: ok")
