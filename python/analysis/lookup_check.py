"""Does the deck network's lookup (core/lookup.py) mean anything? On human winners' card rewards: how often the card the
player took is the option the lookup improves most, against chance, and how the taken cards' improvements compare with
the ones passed over (jack, 2026-10-30, the deck network's step 3: check the signal before training on it).

    .venv/bin/python python/analysis/lookup_check.py python/runs/ironclad-fights-3x128-elites/update_2300.pt \\
        [--decisions ~/sts-data/ironclad/decisions] [--ascension 20] [--n 20000] [--draws 4]

Reads card decisions of won runs at --ascension from the decision shards (human_extract.py; shards in a fixed random
order until --n are found), scores every option with the lookup (each card added, or skipping: this act's boss and the
next act's bosses, --draws openings each), and reports, by act and overall, for three scores (this act's boss, the
next act's bosses, their mean; each the win chance's improvement over skipping):
- the share of decisions where the player's choice is the lookup's best option (skipping counts as an option, at 0),
  against picking an option at random;
- where the player took a card: its improvement against the mean of the cards passed over (paired), and the share
  where it beats every one passed over;
- where the player skipped: the share where no card improves the score (the lookup would skip too), against the same
  share where the player took a card;
- by deck size: the best card's gain, the share of decisions where some card improves the score, and the share where
  the player took one (does the lookup want cards whatever the deck?).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import contextlib
import glob
import gzip
import json
import random
import time

import numpy as np
import torch

from core.lookup import Lookup
from training.train_runs import load


def read_decisions(folder, ascension, n, seed=0):
    paths = sorted(glob.glob(os.path.join(os.path.expanduser(folder), "*.jsonl.gz")))
    random.Random(seed).shuffle(paths)
    out = []
    for p in paths:
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r.get("kind") == "card" and r.get("won") and r.get("ascension") == ascension and r.get("offered"):
                    out.append(r)
        if len(out) >= n:
            break
    return out[:n]


def taken_option(r):
    """The option the player took: -1 skipping, else the index offered (None for anything else, e.g. Singing Bowl)."""
    if r["taken"] == "SKIP":
        return -1
    return r["offered"].index(r["taken"]) if r["taken"] in r["offered"] else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--decisions", default="~/sts-data/ironclad/decisions")
    ap.add_argument("--ascension", type=int, default=20)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=400, help="decisions probed at once (memory: about 64 probe rows each)")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else contextlib.nullcontext
    net, ck = load(a.checkpoint, device)
    net.load_state_dict(ck["net"])
    net.eval()
    lk = Lookup(net, device, amp, draws=a.draws)
    t0 = time.time()
    recs = read_decisions(a.decisions, a.ascension, a.n)
    print(f"{len(recs):,} card decisions of won A{a.ascension} runs read ({time.time() - t0:.0f} s)", flush=True)
    t0 = time.time()
    rows = []  # (act, taken option, options, gains this, gains next, gains mean)
    for at in range(0, len(recs), a.chunk):
        chunk = recs[at:at + a.chunk]
        scores = lk.human_cards([json.dumps(r) for r in chunk])
        for k, sc in scores.items():
            t = taken_option(chunk[k])
            if t is None or t not in sc.options:
                continue
            this = sc.gain([0])[0]
            nxt = sc.gain(list(range(1, sc.win.shape[1])))[0] if sc.win.shape[1] > 1 else this
            rows.append((chunk[k]["act"], sc.options.index(t), len(sc.options), this, nxt, (this + nxt) / 2, len(chunk[k]["deck"])))
        print(f"  {at + len(chunk):,} of {len(recs):,} ({time.time() - t0:.0f} s)", flush=True)

    def report(name, sub):
        if not sub:
            return
        line = [f"{name:8} {len(sub):6} decisions; skipped {100 * np.mean([r[1] == 0 for r in sub]):4.1f}%"]
        chance = np.mean([1 / r[2] for r in sub])
        for label, col in (("this boss", 3), ("next bosses", 4), ("mean", 5)):
            best = np.mean([int(np.argmax(r[col])) == r[1] for r in sub])
            took = [r for r in sub if r[1] > 0]
            diff = [r[col][r[1]] - np.mean([g for j, g in enumerate(r[col]) if j > 0 and j != r[1]]) for r in took if r[2] > 2]
            beats = [r[col][r[1]] > max(g for j, g in enumerate(r[col]) if j > 0 and j != r[1]) for r in took if r[2] > 2]
            no_gain_skip = np.mean([r[col][1:].max() <= 0 for r in sub if r[1] == 0]) if any(r[1] == 0 for r in sub) else float("nan")
            no_gain_take = np.mean([r[col][1:].max() <= 0 for r in took]) if took else float("nan")
            line.append(f"  {label:11}: picks the lookup's best {100 * best:4.1f}% (chance {100 * chance:4.1f}%); a taken card "
                        f"{100 * np.mean(diff):+5.2f} points over the ones passed over, best of them {100 * np.mean(beats):4.1f}%; "
                        f"no card helps: {100 * no_gain_skip:4.1f}% of skips, {100 * no_gain_take:4.1f}% of picks")
        print("\n".join(line))

    print(f"{len(rows):,} decisions scored ({time.time() - t0:.0f} s)")
    report("all", rows)
    for act in (1, 2, 3):
        report(f"act {act}", [r for r in rows if r[0] == act])
    # Does the lookup want cards whatever the deck? By deck size: the best card's gain, the share where some card helps,
    # and the share the players took a card.
    print("by deck size (the mean score): decisions, the best card's gain (points of win chance), some card helps, players took a card")
    for lo, hi in ((0, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 99)):
        sub = [r for r in rows if lo <= r[6] < hi]
        if sub:
            best = [r[5][1:].max() for r in sub]
            print(f"  {lo:2}-{hi - 1:2} cards: {len(sub):6}  {100 * np.mean(best):+6.2f}  {100 * np.mean([b > 0 for b in best]):5.1f}%  "
                  f"{100 * np.mean([r[1] > 0 for r in sub]):5.1f}%")


if __name__ == "__main__":
    main()
