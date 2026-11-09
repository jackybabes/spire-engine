"""How sure a run network is at card rewards (jack, 2026-11-07: "Is the network still super sure of itself when picking up shiv cards?"):
greedy runs, keys free, and at each card reward the probability the policy gives each card and the skip, by part of the run
(search_eval.CARD_BUCKETS: floors 1-8, 9-16, Act 2, Act 3). Prints, for each checkpoint: the mean entropy of a card reward (in bits; a
choice of four is at most 2), the share of rewards where one option takes 90% or more, and for chosen cards their mean probability when offered.

    .venv/bin/python python/analysis/pick_confidence.py python/runs/silent-runs-13/latest.pt python/runs/silent-runs-5/update_1200.pt --runs 256
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import collections
import contextlib
import json

import numpy as np
import torch

from core.fight_model import pick_device
from search.search_eval import CARD_REWARD, CONFIRM, EVAL_SEED, PICK_AT, SCREEN_AT, CardChoices, RunEnv, card_bucket, card_names, load, play

CARDS = ("Blade Dance", "Infinite Blades", "Accuracy", "Cloak and Dagger", "After Image", "Wraith Form", "Footwork", "Noxious Fumes",
         "Catalyst", "Deadly Poison", "Bouncing Flask", "Acrobatics", "Tactician", "Grand Finale", "Backflip", "Piercing Wail", "Dash")
BUCKETS = ("floors 1-8", "floors 9-16", "Act 2", "Act 3")


class Confidence(CardChoices):
    """Writes down each card reward's probabilities as well as the pick."""

    def __init__(self, n):
        super().__init__(n)
        self.rows = []

    def step(self, env, obs, ints, floats, logits, acts, episode, wanted):
        for g in np.flatnonzero(obs["ints"][:, SCREEN_AT] == CARD_REWARD):
            if wanted(g, episode[g]):
                view = json.loads(env.envs.observe(int(g)))
                names = card_names(view["screen"]["cards"])
                p = torch.softmax(logits[g].float().cpu(), -1).numpy()
                self.rows.append((view["floor"], names, [float(p[PICK_AT + k]) for k in range(len(names))], float(p[CONFIRM])))
        super().step(env, obs, ints, floats, logits, acts, episode, wanted)


def report(rows):
    by = collections.defaultdict(list)
    for floor, names, ps, skip in rows:
        b = card_bucket(floor)
        if b is not None:
            by[b].append((names, ps, skip))
    for b, label in enumerate(BUCKETS):
        g = by[b]
        if not g:
            continue
        ent = [-sum(x * np.log2(x) for x in ps + [skip] if x > 0) for _, ps, skip in g]
        sure = [max(ps + [skip]) >= 0.9 for _, ps, skip in g]
        per = collections.defaultdict(list)
        for names, ps, _ in g:
            for n, x in zip(names, ps):
                per[n].append(x)
        print(f"  {label:11} {len(g):5} rewards: entropy {np.mean(ent):.2f} bits, one option at 90%+ in {np.mean(sure):.0%}, skip {np.mean([s for *_, s in g]):.0%}; "
              + ", ".join(f"{c} {np.mean(per[c]):.0%} ({len(per[c])})" for c in CARDS if len(per[c]) >= 5))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--runs", type=int, default=256)
    p.add_argument("--seed-offset", type=int, default=16384)
    p.add_argument("--character", default="silent")
    p.add_argument("--gpu-gb", type=float, default=2.5)
    args = p.parse_args()
    device = pick_device("auto")
    if device.type == "cuda" and args.gpu_gb:
        torch.cuda.set_per_process_memory_fraction(min(args.gpu_gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory, 1.0), 0)
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else contextlib.nullcontext
    for path in args.checkpoints:
        net, ck = load(path, device)
        net.load_state_dict(ck["net"])
        net.eval()
        b = min(args.runs, 256)
        env = RunEnv(b, seed=EVAL_SEED + args.seed_offset, ascension=20, max_steps=8000, character=args.character)
        env.envs.set_free_keys(True)
        env.envs.reset()
        conf = Confidence(b)
        play(net, device, amp, env, lambda g, e: e * b + g < args.runs, None, 0.0, os.path.basename(path), cards=conf)
        print(f"{path} (update {ck.get('update')}), {args.runs} runs, keys free", flush=True)
        report(conf.rows)


if __name__ == "__main__":
    main()
