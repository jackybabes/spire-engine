"""Saves fights played by a checkpoint so they can be watched in the real game.

    .venv/bin/python python/watch/watch_fights.py <checkpoint> <out.json> [encounter ...]

Plays the random-deck fight set of baselines.py greedily and keeps the first fight of each
named encounter (default: a few elites and bosses), won or lost, with its seed, deck and every
move as CommunicationMod commands. tools/trace/playback_fights.py plays them back in the game:
it starts a run on the fight's seed, sets the deck with the `harness` command and starts the
encounter on floor 1, which is the same fight the engine played.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import json
import sys

import numpy as np
import torch

import sts
from core.fight_model import FightNet, masked, pick_device, to_tensors
from core.sts_env import FightEnv

DEFAULT = ["Gremlin Nob", "Hexaghost", "Book of Stabbing", "Champ", "Nemesis"]


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    wanted = sys.argv[3:] or DEFAULT
    unknown = [e for e in wanted if e not in sts.all_encounters()]
    if unknown:
        sys.exit(f"unknown encounters: {unknown}")
    device = pick_device()
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    net = FightNet(d=a["d"], layers=a["layers"], heads=a["heads"]).to(device).eval()
    net.load_state_dict(ck["net"])

    env = FightEnv(128, seed=100_000, encounters=wanted, record=True)
    found = {}
    while len(found) < len(wanted):
        obs = env.observe()
        ints, floats, mask = to_tensors(obs, device)
        with torch.no_grad():
            acts = masked(net(ints, floats)[0], mask).argmax(-1).cpu().numpy()
        for g in np.flatnonzero(env.step(acts)):
            ep = json.loads(env.last_episode(int(g)))
            found.setdefault(ep["encounter"], ep)
    fights = [found[e] for e in wanted]
    with open(out, "w") as f:
        json.dump({"checkpoint": ckpt, "fights": fights}, f, indent=1)
    for ep in fights:
        result = "won" if ep["won"] else "lost"
        print(f"{ep['encounter']:20} {result}, {ep['hp']} HP left, {ep['turns']} turns, {len(ep['moves'])} moves, "
              f"{len(ep['deck'])} cards, seed {ep['seed_string']}")


if __name__ == "__main__":
    main()
