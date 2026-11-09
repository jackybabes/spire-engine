"""Generates a batch of network-played runs for an overnight real-game check on the Mac.

    .venv/bin/python python/watch/soak_runs.py <run checkpoint> [out.json.gz] [--runs 300] [--seed 700000] [--greedy]

The run network plays `--runs` runs of its own character (the checkpoint's; --character to choose
another) on consecutive seeds from `--seed` (away from the training
and eval seeds), sampling its moves (or, with --greedy, taking its top move) so the runs cover
varied choices. Each run is saved in tools/trace/playback.py's format together with how the
engine's run ended; runs that hit the decision limit are left out. tools/trace/soak.sh plays
the batch back in the game and logs every run where the game and the engine differ.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import gzip
import json

import numpy as np
import torch

from core.fight_model import masked, pick_device
from core.sts_env import RunEnv
from training.train_runs import load


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("out", nargs="?", default="python/results/soak/batch.json.gz")
    p.add_argument("--runs", type=int, default=300)
    p.add_argument("--seed", type=int, default=700_000)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--character", default=None, help="whose runs (default: the checkpoint's character)")
    p.add_argument("--ascension-range", type=int, nargs=2, metavar=("LO", "HI"),
                   help="each run's ascension drawn evenly from LO..HI (default: 0)")
    args = p.parse_args()

    device = pick_device()
    net, ck = load(args.checkpoint, device)
    net.load_state_dict(ck["net"])
    net.eval()
    torch.manual_seed(args.seed)

    n = min(args.runs, 512)
    # (the network's own character: before 2026-10-27 every batch was the Ironclad's, whoever played it)
    character = args.character or ck["args"].get("character", "ironclad")
    env = RunEnv(n, seed=args.seed, record=True, max_steps=8000, character=character,
                 ascension_range=tuple(args.ascension_range) if args.ascension_range else None)
    per_game = -(-args.runs // n)
    done = np.zeros(n, dtype=int)
    runs = []
    while (done < per_game).any():
        obs = env.observe()
        ints = torch.from_numpy(obs["ints"]).to(device).long()
        floats = torch.from_numpy(obs["floats"]).to(device)
        mask = torch.from_numpy(obs["mask"]).to(device)
        with torch.no_grad():
            logits = masked(net(ints, floats)[0].float(), mask)
        acts = logits.argmax(-1) if args.greedy else torch.distributions.Categorical(logits=logits).sample()
        status = env.step(acts.cpu().numpy())
        for g in np.flatnonzero(status):
            if done[g] < per_game:
                ep = json.loads(env.last_episode(int(g)))
                if not ep.get("timed_out"):
                    engine = {k: ep[k] for k in ("won", "act", "floor", "hp", "max_hp", "gold", "ascension")}
                    runs.append({"log": ep["log"], "engine": engine})
            done[g] += 1
    runs = runs[:args.runs]
    with gzip.open(args.out, "wt") as f:
        json.dump({"checkpoint": args.checkpoint, "update": ck["update"], "greedy": args.greedy, "runs": runs}, f)
    floors = [r["engine"]["floor"] for r in runs]
    decisions = sum(len(r["log"]["steps"]) for r in runs)
    print(f"{len(runs)} {character} runs (update {ck['update']}, {'greedy' if args.greedy else 'sampled'}): mean floor {np.mean(floors):.1f}, "
          f"reached Act 2 {np.mean([f > 16 for f in floors]):.0%}, Act 3 {np.mean([f > 33 for f in floors]):.0%}, "
          f"won {np.mean([r['engine']['won'] for r in runs]):.0%}; {decisions:,} decisions -> {args.out}")


if __name__ == "__main__":
    main()
