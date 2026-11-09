"""Test positions for the Rust forward pass (`crates/sts-net`): rows from random runs and what
PyTorch's network makes of them, so the Rust network can be checked against it.

    .venv/bin/python python/release/net_fixture.py crates/sts-net/tests/data/tiny       # a tiny random network (committed)
    .venv/bin/python python/release/net_fixture.py /tmp/v1 --weights web/models/ironclad-v1.safetensors

Writes <out>.safetensors (the network; with --weights, a copy's path is recorded instead) and
<out>.json: for each position its encoded row (sparse: [index, value] pairs of the nonzero
entries), the legal action slots, PyTorch's logits for them and the run value. The positions come
from random play of every character, a few of each screen per character, with and without a card
selection in fights (so the Defect's orbs and the Watcher's stances are in).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders
from release.export_release import load_release, signature, write_safetensors, FORMAT  # noqa: E402
from core.run_model import COMBAT, RunNet  # noqa: E402
from core.sts_env import LAYOUT, RunEnv  # noqa: E402

SCREEN_AT = sum(n for _, n in LAYOUT["ints"])  # the run's `screen` field is the first after the fight's
OPTION_AT = sum(n for name, n in LAYOUT["ints"][:[f for f, _ in LAYOUT["ints"]].index("option_card")])


def positions(per_kind=3, seed=0):
    """Encoded rows from random play: up to `per_kind` of each (character, screen, fight selection)
    kind, four times that of fights without a selection."""
    rng = np.random.default_rng(seed)
    rows, counts = [], {}
    for character in ("ironclad", "silent", "defect", "watcher"):
        env = RunEnv(16, seed=seed + len(rows), character=character, ascension=int(rng.integers(0, 21)))
        for _ in range(3000):
            obs = env.observe()
            for i in range(env.n):
                screen = int(obs["ints"][i, SCREEN_AT])
                kind = (character, screen, screen == COMBAT and bool(obs["ints"][i, OPTION_AT] > 0))
                # Fights hold most of the inputs (orbs, stances, moves, hits): four times as many.
                quota = per_kind * (4 if kind[1:] == (COMBAT, False) else 1)
                if counts.get(kind, 0) < quota and obs["mask"][i].any() and rng.random() < 0.05:
                    counts[kind] = counts.get(kind, 0) + 1
                    rows.append((obs["ints"][i].copy(), obs["floats"][i].copy(), obs["mask"][i].copy()))
            env.step([rng.choice(np.flatnonzero(m)) for m in obs["mask"]])
    return rows, counts


def tiny_net(seed=0):
    """A small RunNet with every weight random (zero-initialised ones too), but the deck lookup's inputs, which the
    browser doesn't compute yet (export_release leaves them out: only a network that never used them goes out)."""
    torch.manual_seed(seed)
    net = RunNet(d=16, layers=2, heads=2)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
        for lin in (net.lookup_opt, net.lookup_run):
            lin.weight.zero_()
            lin.bias.zero_()
        net.lookup_pick.weight.zero_()
    return net.eval()


def sparse(a):
    nz = np.flatnonzero(a)
    return [[int(i), a[i].item()] for i in nz]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="path without extension")
    p.add_argument("--weights", help="a release weights file (default: a tiny random network)")
    p.add_argument("--per-kind", type=int, default=3)
    a = p.parse_args()
    if a.weights:
        net, _ = load_release(a.weights)
        weights = os.path.abspath(a.weights)
    else:
        net = tiny_net()
        weights = a.out + ".safetensors"
        meta = {"format": FORMAT, "d": "16", "layers": "2", "heads": "2", **signature(), "source": "net_fixture.py tiny_net(0)"}
        write_safetensors(weights, net.state_dict(), meta)
        weights = os.path.basename(weights)
    rows, counts = positions(a.per_kind)
    ints = torch.from_numpy(np.stack([r[0] for r in rows]).astype(np.int64))
    floats = torch.from_numpy(np.stack([r[1] for r in rows]))
    with torch.no_grad():
        logits, value = net(ints, floats)
    cases = []
    for (i, f, m), lg, v in zip(rows, logits.numpy(), value.numpy()):
        legal = np.flatnonzero(m)
        cases.append({"ints": sparse(i), "floats": sparse(f), "legal": legal.tolist(),
                      "logits": [float(lg[k]) for k in legal], "value": float(v)})
    with open(a.out + ".json", "w") as fh:
        json.dump({"weights": weights, "run_int_len": int(ints.shape[1]), "run_float_len": int(floats.shape[1]), "cases": cases}, fh)
    screens = ["-"] + LAYOUT["screens"]
    print(f"{len(cases)} positions -> {a.out}.json: " + ", ".join(f"{ch} {screens[s]}{' (selection)' if sel else ''} {n}" for (ch, s, sel), n in sorted(counts.items())))


if __name__ == "__main__":
    main()
