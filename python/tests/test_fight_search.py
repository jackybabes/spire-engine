"""Checks for the fight look-ahead (plain asserts, no pytest needed), on the CPU with a tiny untrained network:

    python python/tests/test_fight_search.py

Envs.playout_observe_into writes the same arrays as playout_observe, and FightSearch scores the same with the
network's fixed-shape passes (core.fixed_shapes, uncompiled here; the copies padded to BUCKETS) as with the usual
pass: the same candidates, scores and standard errors."""
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import numpy as np
import torch

from core.fixed_shapes import FixedShapes
from core.run_model import RunNet
from core.sts_env import ACTIONS, RUN_FLOAT_LEN, RUN_INT_LEN, RunEnv
from search.fight_search import FightSearch


def positions(n=16, steps=6):
    """Practice fights a few random moves in, and the games among them with a fight move to make."""
    env, rng = RunEnv(n, seed=8, practice=1.0, practice_decks="mix", character="ironclad"), np.random.default_rng(8)
    for _ in range(steps):
        obs = env.observe()
        env.step([rng.choice(np.flatnonzero(m)) for m in obs["mask"]])
    return env, env.observe()


def main():
    torch.manual_seed(0)
    net = RunNet(d=16, layers=1, heads=2).eval()
    env, obs = positions()
    prac = np.asarray(env.envs.practice_mask(), dtype=bool)
    with torch.no_grad():
        logits = net(torch.from_numpy(obs["ints"]).long(), torch.from_numpy(obs["floats"]), torch.from_numpy(prac))[0].numpy()
    logits = np.where(obs["mask"], logits, -1e9)  # as the rollout hands them over (masked)
    games = np.arange(len(prac))

    # playout_observe_into: the same rows as playout_observe.
    roots = [(int(g), sorted(set(np.flatnonzero(obs["mask"][g])[:2].tolist()))) for g in games if obs["mask"][g].sum() >= 2][:4]
    env.envs.playout_start(roots, 3, 1, 5, 30, True)
    ib, fb, mb, idx = env.envs.playout_observe()
    bufs = (bytearray(len(idx) * RUN_INT_LEN * 4 + 64), bytearray(len(idx) * RUN_FLOAT_LEN * 4 + 64), bytearray(len(idx) * ACTIONS + 8))
    n, idx2 = env.envs.playout_observe_into(*bufs)
    assert n == len(idx) and list(idx2) == list(idx), (n, len(idx))
    assert bytes(bufs[0][:len(ib)]) == ib and bytes(bufs[1][:len(fb)]) == fb and bytes(bufs[2][:len(mb)]) == mb, "the same arrays"
    env.envs.playout_results()

    plain = FightSearch(net, torch.device("cpu"), contextlib.nullcontext, k=4, top=3, value="fight", seed=3)
    fixed = FightSearch(net, torch.device("cpu"), contextlib.nullcontext, k=4, top=3, value="fight", seed=3, fixed=FixedShapes(net, compile=False))
    a = plain.scores(env.envs, obs["ints"], obs["mask"], logits, games, prac)
    b = fixed.scores(env.envs, obs["ints"], obs["mask"], logits, games, prac)
    assert a and set(a) == set(b), (sorted(a), sorted(b))
    for g in a:
        assert list(a[g][0]) == list(b[g][0]), g
        assert np.allclose(a[g][1], b[g][1], atol=1e-4) and np.allclose(a[g][2], b[g][2], atol=1e-4), (g, a[g], b[g])
    assert plain.stats["playout_moves"] == fixed.stats["playout_moves"] > 0
    print(f"fight search: ok ({len(a)} positions searched, {fixed.stats['playouts']} copies, {fixed.stats['playout_moves']} moves; "
          f"the same with fixed shapes and reused buffers)")


if __name__ == "__main__":
    main()
