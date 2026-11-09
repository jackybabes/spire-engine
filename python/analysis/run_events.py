"""What a run checkpoint leaves on reward screens and picks at events, on the fixed eval runs (greedy).

    .venv/bin/python python/analysis/run_events.py <run checkpoint> [--runs 256] [--ascension 0]
    .venv/bin/python python/analysis/run_events.py <run checkpoint> --force [--runs 512]

Reward screens: each time the network leaves one (Proceed), the rewards still on it (gold, potions
with a slot free or not, relics, cards, keys), and how much gold. Events: for each event, how
often each option was picked (by its label, the engine's short description), with the runs'
outcome after it. Writes the counts as JSON next to the checkpoint.

With --force: does taking the risky option at events pay, for this network? Plays the same runs
once with the network's own choices, then once per event in TAKE (and once with all of them),
picking that option whenever it is offered (the network decides everything else, the event's
later screens included), and prints each one's win rate and floor against the network's own.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import collections
import json

import numpy as np
import torch

from core.fight_model import masked, pick_device
from core.sts_env import LAYOUT, RunEnv
from training.train_runs import EVAL_SEED, load

# The option to force at each event: the risk human players usually take (HP, a curse, a fight,
# gold) for something that makes the deck stronger.
TAKE = {
    "Golden Idol": "take the idol", "Scrap Ooze": "reach inside", "Shining Light": "enter", "Cursed Tome": "read",
    "Dead Adventurer": "search", "Mysterious Sphere": "open the sphere", "World of Goop": "gather gold",
    "Duplicator": "pray", "Golden Wing": "pray", "Masked Bandits": "fight", "Liars Game": "agree",
    "Fountain of Cleansing": "drink",
}

SCREENS = ["-"] + LAYOUT["screens"]
PICK, _ = LAYOUT["actions"]["pick"]
CONFIRM = LAYOUT["actions"]["confirm"]


def play(net, device, runs, ascension, character, take=None):
    """The fixed runs, greedily; at an event in `take` (event -> label prefix), that option
    whenever it is offered. Returns the run summaries and how many times an option was forced."""
    env = RunEnv(runs, seed=EVAL_SEED, max_steps=8000, ascension=ascension, character=character)
    event_screen = SCREENS.index("event")
    res, forced = [None] * runs, 0
    while any(r is None for r in res):
        obs = env.observe()
        with torch.no_grad():
            lg, _ = net(torch.from_numpy(obs["ints"]).to(device).long(), torch.from_numpy(obs["floats"]).to(device))
        acts = masked(lg.float(), torch.from_numpy(obs["mask"]).to(device)).argmax(-1).cpu().numpy()
        for g in np.flatnonzero(obs["i_screen"][:, 0] == event_screen) if take else []:
            if res[g] is not None:
                continue
            view = json.loads(env.envs.observe(int(g)))["screen"]
            want = take.get(view["id"])
            for j, o in enumerate(view["options"]):
                if want and o["label"].startswith(want) and obs["mask"][g, PICK + j]:
                    forced += acts[g] != PICK + j
                    acts[g] = PICK + j
                    break
        for g in np.flatnonzero(env.step(acts)):
            if res[g] is None:
                res[g] = json.loads(env.last_episode(int(g)))
    return res, forced


def force(net, device, runs, ascension, character):
    def line(name, res, forced=None):
        won, floor = np.mean([r["won"] for r in res]), np.mean([r["floor"] for r in res])
        extra = f" ({forced} choices changed)" if forced is not None else ""
        print(f"  {name:24} won {won:6.1%}  floor {floor:5.1f}{extra}", flush=True)
        return won

    base, _ = play(net, device, runs, ascension, character)
    w0 = line("network's own", base)
    for name, take in [("all of them", TAKE)] + [(k, {k: v}) for k, v in TAKE.items()]:
        res, forced = play(net, device, runs, ascension, character, take)
        w = line(name, res, forced)
        print(f"  {'':24} {w - w0:+.1%} against the network's own", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--runs", type=int, default=256)
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("--force", action="store_true", help="win rates with the risky option forced at each event (TAKE)")
    a = p.parse_args()
    device = pick_device()
    net, ck = load(a.checkpoint, device)
    net.load_state_dict(ck["net"])
    net.eval()
    character = ck["args"].get("character", "ironclad")
    if a.force:
        print(f"{a.checkpoint} (update {ck.get('update')}), {character} A{a.ascension}, {a.runs} runs", flush=True)
        force(net, device, a.runs, a.ascension, character)
        return
    env = RunEnv(a.runs, seed=EVAL_SEED, max_steps=8000, ascension=a.ascension, character=character)
    rewards_screen, event_screen = SCREENS.index("rewards"), SCREENS.index("event")

    left = collections.Counter()           # reward kind -> times left behind
    gold_left, gold_left_runs = [], set()  # amounts; runs that left any
    leaves = 0
    events = collections.defaultdict(collections.Counter)  # event -> label -> picks
    labels_seen = collections.defaultdict(set)              # event -> every label offered
    res = [None] * a.runs
    while any(r is None for r in res):
        obs = env.observe()
        with torch.no_grad():
            lg, _ = net(torch.from_numpy(obs["ints"]).to(device).long(), torch.from_numpy(obs["floats"]).to(device))
        acts = masked(lg.float(), torch.from_numpy(obs["mask"]).to(device)).argmax(-1).cpu().numpy()
        for g in range(a.runs):
            if res[g] is not None:
                continue
            screen = SCREENS[obs["i_screen"][g, 0]]
            if screen == "rewards" and acts[g] == CONFIRM:
                view = json.loads(env.envs.observe(g))
                items = view["screen"]["items"]
                leaves += 1
                free_slot = any(x is None for x in view.get("potions", [None]))
                for it in items:
                    k = it["type"]
                    if k == "potion":
                        k = "potion (slot free)" if free_slot else "potion (slots full)"
                    left[k] += 1
                    if it["type"] in ("gold", "stolen_gold"):
                        gold_left.append(it["amount"])
                        gold_left_runs.add(g)
            elif screen == "event" and PICK <= acts[g] < CONFIRM:
                view = json.loads(env.envs.observe(g))["screen"]
                opts = view["options"]
                j = acts[g] - PICK
                for o in opts:
                    labels_seen[view["id"]].add(o["label"])
                if j < len(opts):
                    events[view["id"]][opts[j]["label"]] += 1
        for g in np.flatnonzero(env.step(acts)):
            if res[g] is None:
                res[g] = json.loads(env.last_episode(int(g)))

    print(f"{a.checkpoint} (update {ck.get('update')}), {character} A{a.ascension}: {a.runs} runs, "
          f"won {np.mean([r['won'] for r in res]):.1%}, floor {np.mean([r['floor'] for r in res]):.1f}")
    print(f"REWARD SCREENS left: {leaves}; rewards left behind: {dict(left)}")
    print(f"  gold left {len(gold_left)} times in {len(gold_left_runs)} of {a.runs} runs, "
          f"{sum(gold_left)} gold in all ({sum(gold_left) / a.runs:.1f} a run)")
    print("EVENTS (picks by option)")
    for ev in sorted(events, key=lambda e: -sum(events[e].values())):
        n = sum(events[ev].values())
        picks = ", ".join(f"{lab!r} {c}" for lab, c in events[ev].most_common())
        print(f"  {ev} ({n}): {picks}")
    out = os.path.join(os.path.dirname(a.checkpoint), f"events_{ck.get('update')}_a{a.ascension}.json")
    with open(out, "w") as f:
        json.dump({"update": ck.get("update"), "runs": a.runs, "left": left, "gold_left": gold_left,
                   "gold_left_runs": len(gold_left_runs), "events": events, "labels": {k: sorted(v) for k, v in labels_seen.items()}}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
