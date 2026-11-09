"""Checks for train_runs.Collector's rollouts (plain asserts, no pytest needed), on the CPU with a tiny
untrained network:

    python python/tests/test_rollout.py

The rollout, with the usual pass and with the fixed-shape passes (uncompiled here), fills the
rollout buffers: every move a legal one, the moves' log-probabilities, values, advantages and returns
finite, the practice flags those of the games, the episodes that ended counted, and where its time
went reported. With --retry-losses and --retry-temp, games replaying a lost fight sample from the
flattened policy and keep the policy's own log-probabilities apart ("logp_pi")."""
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import numpy as np
import torch

from core.fixed_shapes import FixedShapes
from core.run_model import RunNet
from training.train_runs import Collector, parse_args, rollout_buffers


def rollout(fixed):
    args = parse_args(["--name", "test", "--character", "defect", "--practice", "1", "--practice-decks", "mix", "--envs", "16",
                       "--steps", "24", "--eval-runs", "0", "--scale-rewards"] + (["--fixed-shapes"] if fixed else []))
    torch.manual_seed(0)
    net = RunNet(d=16, layers=1, heads=2)
    policy = FixedShapes(net, compile=False) if fixed else net
    col = Collector(args, torch.device("cpu"), contextlib.nullcontext, 0)
    out = rollout_buffers(args.steps, args.envs, "cpu")
    ro = col.collect(0, policy, out)
    T, N = args.steps, args.envs
    acts = out["act"]
    assert ((acts >= 0) & (acts < out["mask"].shape[-1])).all()
    legal = out["mask"].gather(2, acts[..., None]).squeeze(-1)
    assert legal.all(), f"{int((~legal).sum())} illegal moves"
    for k in ("logp", "val", "adv", "ret"):
        assert torch.isfinite(out[k]).all() and out[k].abs().sum() > 0, k
    assert (out["logp"] <= 0).all()
    assert out["prac"].all(), "every game a practice fight"
    assert (out["ints"].abs().sum(-1) > 0).all(), "every row filled"
    assert len(ro["practice_now"]) > 0 and np.isfinite(ro["reward_sum"])
    # The next rollout carries on from where this one left off.
    ro2 = col.collect(1, policy, out)
    assert len(ro2["practice_now"]) > 0
    return ro


def replays():
    args = parse_args(["--name", "test", "--character", "defect", "--practice", "1", "--practice-decks", "mix", "--envs", "16",
                       "--steps", "64", "--eval-runs", "0", "--retry-losses", "8", "--retry-temp", "2"])
    torch.manual_seed(0)
    net = RunNet(d=16, layers=1, heads=2)
    col = Collector(args, torch.device("cpu"), contextlib.nullcontext, 0)
    out = rollout_buffers(args.steps, args.envs, "cpu")
    ro = col.collect(0, net, out)
    replayed = [e for e in ro["practice_now"] if e.get("retry")]
    assert replayed, "some lost fights replayed"
    assert torch.isfinite(out["logp_pi"]).all() and (out["logp_pi"] <= 0).all()
    same = torch.isclose(out["logp"], out["logp_pi"])
    assert same.any() and (~same).any(), "first tries sampled from the policy, replays from the flattened one"
    print(f"replays: ok ({len(replayed)} replays ended; {(~same).float().mean():.0%} of moves sampled flattened)")


def main():
    replays()
    a = rollout(False)
    b = rollout(True)
    assert set(b["phase_s"]) == {"to_gpu", "network", "engine", "observe", "glue"}
    assert len(a["practice_now"]) == len(b["practice_now"]), "the fixed passes play as the usual one"
    print(f"rollouts: ok ({len(a['practice_now'])} fights ended, the same with fixed shapes)")


if __name__ == "__main__":
    main()
