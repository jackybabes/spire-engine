"""The decks a run network builds, against human winners' (is it building good Defect decks?).

    .venv/bin/python python/analysis/deck_report.py <checkpoint> [--runs 256] [--ascension 0] \
        [--human ~/sts-data/defect/stats.json]

Plays --runs whole runs greedily (fixed seeds) and compares the decks they end with to the won
decks in the human data (`human_extract.py`'s stats.json): the deck size, the Strikes and
Defends left, the cards upgraded, and the share of decks holding each card (a card's presence;
the network's runs that died are counted apart, since a deck that dies early is small). Prints
the cards the network holds most and least often compared with human winners, and the human
winners' most common cards it rarely holds.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import collections
import contextlib
import json
import statistics as st

import numpy as np
import torch
import torch.nn.functional as F

import sts
from core.fight_model import masked
from core.sts_env import RunEnv
from training.train_runs import EVAL_SEED, load

BASICS = ("Strike", "Defend")


def play(net, device, amp, runs, ascension, character, batch=128):
    """`runs` whole runs, greedily, on the eval's fixed seeds: their episode summaries (with decks)."""
    net.eval()
    env = RunEnv(min(batch, runs), seed=EVAL_SEED, ascension=ascension, max_steps=8000, character=character, record=True)
    n = len(env.envs)
    done, out = np.zeros(n, dtype=int), []
    per_game = -(-runs // n)
    while (done < per_game).any():
        obs = env.observe()
        ints = torch.from_numpy(obs["ints"]).to(device).long()
        floats = torch.from_numpy(obs["floats"]).to(device)
        mask = torch.from_numpy(obs["mask"]).to(device)
        with torch.no_grad(), amp():
            logits, _ = net(ints, floats)
        acts = F.log_softmax(masked(logits.float(), mask), -1).argmax(-1).cpu().numpy()
        for g in np.flatnonzero(env.step(acts)):
            if done[g] < per_game:
                out.append(json.loads(env.last_episode(int(g))))
            done[g] += 1
    return out[:runs]


def presence(decks):
    """The share of `decks` holding each card (by name, upgrades folded in)."""
    c = collections.Counter()
    for d in decks:
        c.update({x.rstrip("+") for x in d})
    return {k: v / max(len(decks), 1) for k, v in c.items()}


def describe(name, decks):
    if not decks:
        return f"{name}: none"
    size = [len(d) for d in decks]
    basics = [sum(x.rstrip("+") in BASICS for x in d) for d in decks]
    up = [sum(x.endswith("+") for x in d) for d in decks]
    return (f"{name}: {len(decks)} decks, size median {st.median(size):.0f} (25-75th {np.percentile(size, 25):.0f}-{np.percentile(size, 75):.0f}), "
            f"Strikes and Defends left {st.mean(basics):.1f}, upgraded {st.mean(up):.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--runs", type=int, default=256)
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("--human", default=os.path.expanduser("~/sts-data/defect/stats.json"))
    p.add_argument("--out", default=None, help="JSON report (default: next to the checkpoint)")
    p.add_argument("--cards", default="Hologram,Defragment,Capacitor,Loop,Recursion,Storm,Electrodynamics,Echo Form",
                   help="cards to report on their own: the share of decks holding one and holding it upgraded")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else contextlib.nullcontext
    net, ck = load(a.checkpoint, device)
    net.load_state_dict(ck["net"])
    character = ck["args"].get("character", "ironclad")
    eps = play(net, device, amp, a.runs, a.ascension, character)
    won = [e["deck"] for e in eps if e["won"]]
    lost = [e["deck"] for e in eps if not e["won"]]
    floors = [e["floor"] for e in eps]
    print(f"{a.checkpoint} (update {ck.get('update')}), {character} at A{a.ascension}: won {len(won)} of {len(eps)}, "
          f"floor median {st.median(floors):.0f}, reached Act 2 {np.mean([e['act'] >= 2 for e in eps]):.0%}, Act 3 {np.mean([e['act'] >= 3 for e in eps]):.0%}")

    layout = sts.layout()
    name_of = dict(zip(layout["card_java_ids"], layout["card_names"]))
    h = json.load(open(a.human))
    h_won = h["won"]
    # Several ids share a name (each character's Strike and Defend; other colours come in by
    # Prismatic Shard): the largest share stands for the name.
    human = {}
    for k, v in h["deck_won"].items():
        n = name_of.get(k, k)
        human[n] = max(human.get(n, 0.0), v / h_won)
    sizes = [int(k) for k, v in h["deck_size_won"].items() for _ in range(min(v, 100000))]
    print(f"human winners: {h_won:,} decks, size median {st.median(sizes):.0f} (25-75th {np.percentile(sizes, 25):.0f}-{np.percentile(sizes, 75):.0f})")
    print(describe("network, won", won))
    print(describe("network, lost", lost))
    print(describe("network, all", won + lost))

    mine = presence(won if len(won) >= 30 else won + lost)
    which = "won" if len(won) >= 30 else "all"
    rows = sorted(set(mine) | set(human), key=lambda c: -(mine.get(c, 0) - human.get(c, 0)))
    fmt = lambda c: f"{c} {mine.get(c, 0):.0%} vs {human.get(c, 0):.0%}"  # noqa: E731
    print(f"held far more often than by human winners (network {which} vs human won): " + ", ".join(fmt(c) for c in rows[:12]))
    print("held far less often: " + ", ".join(fmt(c) for c in rows[::-1][:12]))
    decks = won + lost
    for c in [x.strip() for x in a.cards.split(",") if x.strip()]:
        held = np.mean([any(x.rstrip("+") == c for x in d) for d in decks]) if decks else 0
        up = np.mean([any(x == c + "+" for x in d) for d in decks]) if decks else 0
        print(f"  {c}: network decks holding it {held:.0%} (upgraded {up:.0%}), human winners {human.get(c, 0):.0%}")
    common = sorted(human, key=lambda c: -human[c])[:25]
    print("human winners' 25 most held: " + ", ".join(fmt(c) for c in common))
    report = {"checkpoint": a.checkpoint, "update": ck.get("update"), "ascension": a.ascension, "runs": len(eps), "won": len(won),
              "floor_median": st.median(floors), "network_presence": mine, "human_presence": human,
              "network_deck_size": [len(d) for d in won + lost]}
    out = a.out or os.path.join(os.path.dirname(a.checkpoint), f"deck_report_{ck.get('update')}_a{a.ascension}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
