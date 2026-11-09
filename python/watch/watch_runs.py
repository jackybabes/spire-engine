"""Saves whole runs played by a run checkpoint so they can be watched in the real game or the
browser viewer.

    .venv/bin/python python/watch/watch_runs.py <run checkpoint> [out.json] [runs to try] [--character silent] [--ascension 0]

Plays `runs to try` runs (default 64) greedily from the fixed eval seeds and saves the one that
got furthest (highest floor; a win beats everything) in tools/trace/playback.py's format, the
same as `sts-cli run --save`. Watch it on the Mac with tools/trace/watch.sh <out.json>, or in the
browser (web/serve.sh; a file in python/results/watch/ is at /?run=/runs/<name>).

With --until-heart N, it keeps playing batches of `runs to try` on new seeds (the next batch's
first seed follows the last) until a run beats the Heart, at most N batches, and saves that one
(else the furthest).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import json

import numpy as np
import torch

from core.fight_model import masked, pick_device
from core.sts_env import RunEnv
from training.train_runs import EVAL_SEED, load


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("out", nargs="?", default="python/results/watch/run.json")
    p.add_argument("tries", nargs="?", type=int, default=64)
    p.add_argument("--character", default="ironclad", choices=["ironclad", "silent", "defect", "watcher"])
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("--until-heart", type=int, default=0, metavar="N", help="batches on new seeds until a run beats the Heart (at most N)")
    a = p.parse_args()
    ckpt, out, tries = a.checkpoint, a.out, a.tries
    device = pick_device()
    net, ck = load(ckpt, device)
    net.load_state_dict(ck["net"])
    net.eval()

    def play(seed):
        env = RunEnv(tries, seed=seed, record=True, max_steps=8000, ascension=a.ascension, character=a.character)
        results = [None] * tries
        while any(r is None for r in results):
            obs = env.observe()
            ints = torch.from_numpy(obs["ints"]).to(device).long()
            floats = torch.from_numpy(obs["floats"]).to(device)
            mask = torch.from_numpy(obs["mask"]).to(device)
            with torch.no_grad():
                logits, _ = net(ints, floats)
            status = env.step(masked(logits.float(), mask).argmax(-1).cpu().numpy())
            for g in np.flatnonzero(status):
                if results[g] is None:
                    results[g] = json.loads(env.last_episode(int(g)))
        return results

    results = []
    for k in range(max(a.until_heart, 1)):
        results += play(EVAL_SEED + k * tries)
        if a.until_heart:
            hearts = sum(r["won"] and r["act"] >= 4 for r in results)
            print(f"  batch {k + 1}: {len(results)} runs, {sum(r['act'] >= 4 for r in results)} reached Act 4, {hearts} beat the Heart", flush=True)
            if hearts:
                break
    tries = len(results)
    # A win in Act 4 (the Heart) is the furthest a run goes.
    best = max(results, key=lambda r: (r["won"], r["act"], r["floor"], r["hp"]))
    floors = [r["floor"] for r in results]
    wins = sum(r["won"] for r in results)
    print(f"{ckpt} (update {ck['update']}), {a.character} A{a.ascension}: {tries} runs, {wins} won, mean floor {np.mean(floors):.1f}, best floor {best['floor']} "
          f"(act {best['act']}, {'won' if best['won'] else 'died'}), seed {best['log']['seed_string']}, {len(best['log']['steps'])} decisions")
    with open(out, "w") as f:
        json.dump(best["log"], f, indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
