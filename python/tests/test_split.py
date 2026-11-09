"""Checks for the split (notes/split-networks-plan.md, D1-D2; plain asserts, no pytest needed):

    python python/tests/test_split.py

core.ppo.choice_gae, the deck network's advantages over the choice steps only: equal to a sum
written out by hand, and to GAE itself when every step is a choice. core.split.SplitPolicy: on
positions from random Watcher runs, fight rows get the fight network's logits and value and the
rest the deck network's, the same through the fixed passes as through the networks themselves.
train_runs.evaluate with the same network fighting (`fight_net`) plays the same runs as without it,
and another fight network plays other runs. The split the other way round (train_runs.py --deck-net): train() reaches the
fight network only, and a rollout's advantages are on the fight rows only."""
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import numpy as np
import torch

from core.fixed_shapes import FixedShapes
from core.ppo import choice_gae, gae
from core.run_model import RunNet
from core.split import SplitPolicy, fight_rows
from core.sts_env import RunEnv
from training.train_runs import Collector, evaluate, parse_args, rollout_buffers


def brute_choice_gae(rew, val, done, next_val, choice, gamma, lam):
    """The same, written out: each choice's transition to the next choice of its episode (or the
    rollout's end, bootstrapped with next_val), then GAE over those transitions."""
    T, N = rew.shape
    adv = torch.zeros(T, N, dtype=torch.float64)
    for n in range(N):
        steps = [t for t in range(T) if choice[t, n]]
        trans = []  # (t, reward, discount, next value)
        for t in steps:
            r, d, j = 0.0, 1.0, t
            while True:
                r += d * float(rew[j, n])
                d *= gamma
                if done[j, n]:
                    d, nv = 0.0, 0.0
                    break
                j += 1
                if j == T:
                    nv = float(next_val[n])
                    break
                if choice[j, n]:
                    nv = float(val[j, n])
                    break
            trans.append((t, r, d, nv))
        last = 0.0
        for t, r, d, nv in reversed(trans):
            delta = r + d * nv - float(val[t, n])
            last = delta + lam * d * last
            adv[t, n] = last
    return adv


def test_choice_gae():
    g = torch.Generator().manual_seed(0)
    T, N, gamma, lam = 40, 7, 0.97, 0.9
    rew, val, next_val = torch.randn(T, N, generator=g), torch.randn(T, N, generator=g), torch.randn(N, generator=g)
    done = (torch.rand(T, N, generator=g) < 0.08).float()
    choice = torch.rand(T, N, generator=g) < 0.3
    adv, ret = choice_gae(rew, val, done, next_val, choice, gamma, lam)
    want = brute_choice_gae(rew, val, done, next_val, choice, gamma, lam)
    assert torch.allclose(adv.double(), want, atol=1e-5), (adv - want).abs().max()
    assert torch.allclose(ret[choice], (adv + val)[choice]) and (adv[~choice] == 0).all() and (ret[~choice] == 0).all()
    # Every step a choice: plain GAE.
    every = torch.ones(T, N, dtype=torch.bool)
    adv, ret = choice_gae(rew, val, done, next_val, every, gamma, lam)
    a2, r2 = gae(rew, val, done, next_val, gamma, lam)
    assert torch.allclose(adv, a2, atol=1e-5) and torch.allclose(ret, r2, atol=1e-5)


def positions(n_games=64, steps=40):
    env = RunEnv(n_games, seed=5, character="watcher")
    ints, floats, mask = [], [], []
    for k in range(steps):
        obs = env.observe()
        ints.append(obs["ints"].copy())
        floats.append(obs["floats"].copy())
        mask.append(obs["mask"].copy())
        env.envs.step_random(200 + k)
    return (torch.from_numpy(np.concatenate(ints).astype(np.int64)), torch.from_numpy(np.concatenate(floats)),
            torch.from_numpy(np.concatenate(mask)))


def test_split_policy():
    torch.manual_seed(0)
    ints, floats, mask = positions()
    practice = torch.zeros(len(ints), dtype=torch.bool)
    combat = fight_rows(ints)
    assert combat.any() and (~combat).any(), "random runs reach fights and choices"
    fight, deck = RunNet(d=32, layers=1, heads=2).eval(), RunNet(d=32, layers=1, heads=2).eval()
    with torch.no_grad():
        lf, vf = fight(ints, floats, practice)
        ld, vd = deck(ints, floats, practice)
        ls, vs = SplitPolicy(fight, deck)(ints, floats, practice)
        assert torch.equal(ls[combat], lf[combat]) and torch.equal(ls[~combat], ld[~combat])
        assert torch.equal(vs[combat], vf[combat]) and torch.equal(vs[~combat], vd[~combat])
        # The fixed passes (uncompiled here) give the same, each row once.
        lx, vx = SplitPolicy(FixedShapes(fight, compile=False), FixedShapes(deck, compile=False))(ints, floats, practice)
        print(f"fixed passes against the networks: logits of legal moves within {(lx - ls)[mask].abs().max():.1e}, "
              f"values within {(vx - vs).abs().max():.1e}")
        assert torch.allclose(lx[mask], ls[mask], atol=1e-4) and torch.allclose(vx, vs, atol=1e-4)


def test_evaluate():
    torch.manual_seed(0)
    net, other = RunNet(d=32, layers=1, heads=2), RunNet(d=32, layers=1, heads=2)
    kw = dict(runs=8, amp=contextlib.nullcontext, ascension=0, batch=8, max_steps=600, character="watcher")
    alone = evaluate(net, torch.device("cpu"), **kw)
    same = evaluate(net, torch.device("cpu"), fight_net=net, **kw)
    assert alone == same, (alone, same)
    mixed = evaluate(net, torch.device("cpu"), fight_net=other, **kw)
    assert mixed != alone
    print(f"8 runs: floor {alone['floor']:.1f} alone and with itself fighting, {mixed['floor']:.1f} with another network fighting")


def test_deck_net():
    torch.manual_seed(1)
    fight, deck = RunNet(d=16, layers=1, heads=2), RunNet(d=16, layers=1, heads=2)
    policy = SplitPolicy(fight, deck, trains="fight")
    policy.eval()
    policy.train()
    assert fight.training and not deck.training, "only the fight network trains"
    args = parse_args(["--name", "test", "--character", "watcher", "--envs", "16", "--steps", "48", "--eval-runs", "0",
                       "--practice", "0.25", "--practice-decks", "mix"])
    args.deck_net = "fixed"  # (the rollout only asks whether it is set)
    col = Collector(args, torch.device("cpu"), contextlib.nullcontext, 0)
    out = rollout_buffers(args.steps, args.envs, "cpu")
    col.collect(0, policy.eval(), out)
    combat = fight_rows(out["ints"].reshape(-1, out["ints"].shape[-1])).reshape(args.steps, args.envs)
    assert combat.any() and (~combat).any(), "fights and choices"
    assert (out["adv"][~combat] == 0).all() and (out["ret"][~combat] == 0).all(), "no advantage on the choices"
    assert (out["adv"][combat] != 0).any()
    print(f"--deck-net: advantages on {int(combat.sum())} fight rows of {combat.numel()}, none on the choices")


if __name__ == "__main__":
    test_choice_gae()
    test_split_policy()
    test_evaluate()
    test_deck_net()
    print("ok")
