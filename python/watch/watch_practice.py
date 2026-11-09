"""Saves practice fights played by a run checkpoint so they can be watched in the real game.

    .venv/bin/python python/watch/watch_practice.py <checkpoint> <out.json> [--fights 12] [--acts 1,2,3]
    tools/trace/watch.sh <out.json>

Plays the checkpoint's practice fights (its run's --practice-decks: a deck grown from the starting
one, an engine deck or a random deck, partway through an act) greedily, set up as the modded
game's `harness` command sets them up (`practice::harness_fight` in crates/sts-py: a new run on
the fight's seed, its relics, deck, HP and potions, the encounter in a monster room on floor 1),
and keeps --fights of them, spread over the acts, won or lost, with every move as
CommunicationMod commands (picked at random from each game's first fight, so short fights aren't
favoured). tools/trace/playback_fights.py plays them back. A0-10 only (the
harness can't set up an A11+ game's two potion slots); Act 4 is left out by default (no Act 4
fight has been checked against the game).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import json
import random

import numpy as np
import torch

from core.fight_model import masked
from core.sts_env import RunEnv
from training.train_runs import deck_kind, load


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("out")
    p.add_argument("--fights", type=int, default=12, help="fights to keep, spread over --acts")
    p.add_argument("--acts", default="1,2,3", help="acts to take fights from (4 is unchecked against the game)")
    p.add_argument("--ascension-range", type=int, nargs=2, default=(0, 10), metavar=("LO", "HI"), help="A0-10 at most")
    p.add_argument("--decks", default=None, help="deck kinds: mix, grown, engine or random (default: the run's)")
    p.add_argument("--seed", type=int, default=900_000)
    a = p.parse_args()
    if a.ascension_range[1] > 10:
        sys.exit("--ascension-range: A0-10 at most (the harness can't set up an A11+ game's potion slots)")
    acts = [int(x) for x in a.acts.split(",")]

    torch.set_num_threads(4)
    net, ck = load(a.checkpoint, torch.device("cpu"))
    net.load_state_dict(ck["net"])
    net.eval()
    args = ck["args"]
    character = args.get("character", "ironclad")
    decks = a.decks or args.get("practice_decks") or "mix"
    want = {act: a.fights // len(acts) + (i < a.fights % len(acts)) for i, act in enumerate(acts)}
    games = max(64, 8 * a.fights)
    env = RunEnv(games, seed=a.seed, practice=1.0, max_steps=2000, character=character, practice_decks=decks,
                 ascension_range=tuple(a.ascension_range), record=True, harness=True)

    first = [None] * games
    while any(ep is None for ep in first):
        obs = env.observe()
        ints = torch.from_numpy(obs["ints"]).long()
        floats = torch.from_numpy(obs["floats"])
        mask = torch.from_numpy(obs["mask"])
        with torch.no_grad():
            logits, _ = net(ints, floats, torch.ones(len(ints), dtype=torch.bool))
        for g in np.flatnonzero(env.step(masked(logits.float(), mask).argmax(-1).numpy())):
            if first[g] is None:
                first[g] = json.loads(env.last_episode(int(g)))
    pick = random.Random(a.seed)
    fights = []
    for act in acts:
        pool = [ep for ep in first if ep["act"] == act and not ep["timed_out"]]
        fights += pick.sample(pool, min(want[act], len(pool)))
    with open(a.out, "w") as f:
        json.dump({"checkpoint": a.checkpoint, "update": ck.get("update"), "fights": fights}, f, indent=1)
    for ep in fights:
        result = f"won, {ep['hp']} HP left" if ep["won"] else "lost"
        print(f"Act {ep['act']} A{ep['ascension']:<2} {deck_kind(ep['practice']):6} {ep['encounter']:22} {result}, "
              f"{len(ep['moves'])} moves, seed {ep['seed_string']}")
    print(f"{len(fights)} fights from update {ck.get('update')} in {a.out}; watch them with tools/trace/watch.sh {a.out}")


if __name__ == "__main__":
    main()
