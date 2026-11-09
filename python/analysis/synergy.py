"""What-if synergies: how one card in the deck (or one relic) changes a network's card picks.

    .venv/bin/python python/analysis/synergy.py <weights or run checkpoint> [--runs 160] [--ascensions 10,20] \
        [--out table.json] [--top 25] [--show "card: Corruption>Feel No Pain,relic: Snecko Eye>Bludgeon"]

1. Plays greedy runs at each ascension (fixed seeds) and keeps every card-reward position: the
   encoded input, the legal moves and the offered cards.
2. For each thing X, a card added to the deck (the character's own and colorless ones) or a
   relic gained (shared and the character's own), changes only that in every position (the
   deck's size goes up with a card; a relic already held is skipped) and runs the network again.
3. The synergy of X with an offered card Y: how much the log-probability of picking Y moved,
   averaged over the positions where Y was offered (pairs seen fewer than --min-positions times are
   left out). +1 is about 2.7 times as likely, +2 about 7 times.

Prints the strongest synergies and anti-synergies, the pairs asked for with --show, and how many
pairs have an effect over 1.0 (a measure of how much the network knows about combining things).
Caveats: one change at a time, on the network's own positions; a relic can stand for the part of
the run it usually comes with (Mark of the Bloom: late Act 3, after Mind Bloom).

A network released for an older encoding (ironclad-v1, silent-v1: encoding 1) runs only with the
code it was released from: a worktree at that commit (`git worktree add --detach <dir> 41bbd09`),
its `sts` module built there (`python/build.sh`), and this script copied in and run there.
First run on defect-v1 and ironclad-v1 2026-10-26 (6,000 positions each, about 8 minutes on a
Mac): the Ironclad network knows its deck plans (869 pairs over 1.0: Dead Branch with Havoc and
Feel No Pain, Barricade with Entrench and Body Slam, Snecko Eye with expensive cards), the Defect
network much less (126: Claw with All for One, Heatsinks with Storm, but nothing between Focus and
the orb cards).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import json
import re
import statistics

import numpy as np
import torch

import sts
from core.fight_model import masked, pick_device
from core.run_model import RUN_FLOAT_AT, RUN_INT_AT
from core.sts_env import LAYOUT, RunEnv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCREENS = ["-"] + LAYOUT["screens"]
PICK, _ = LAYOUT["actions"]["pick"]
DECK_SIZE = 6  # the deck's size in run_scalars (crates/sts-encode/src/encode_run.rs)
COLOR = {"ironclad": "Red", "silent": "Green", "defect": "Blue", "watcher": "Purple"}
SEEDS = 500_000  # the positions' runs: seeds from SEEDS + 10,000 x ascension (not the evals')


def load_network(path, device):
    """A release weights file or a run checkpoint -> (network in eval mode, character)."""
    if path.endswith(".safetensors"):
        from release.export_release import load_release
        net, meta = load_release(path, device)
        return net, meta.get("character", "ironclad")
    from training.train_runs import load
    net, ck = load(path, device)
    net.load_state_dict(ck["net"])
    return net.eval(), ck["args"].get("character", "ironclad")


def positions(net, device, runs, asc, character):
    """Every card-reward position of `runs` greedy runs: (ints, floats, masks, offered names)."""
    env = RunEnv(runs, seed=SEEDS + 10_000 * asc, max_steps=8000, ascension=asc, character=character)
    card_screen = SCREENS.index("card_reward")
    ints, floats, masks, offered = [], [], [], []
    done = [False] * runs
    while not all(done):
        obs = env.observe()
        with torch.no_grad():
            lg, _ = net(torch.from_numpy(obs["ints"]).to(device).long(), torch.from_numpy(obs["floats"]).to(device))
        acts = masked(lg.float(), torch.from_numpy(obs["mask"]).to(device)).argmax(-1).cpu().numpy()
        for g in np.flatnonzero(obs["i_screen"][:, 0] == card_screen):
            if not done[g]:
                ints.append(obs["ints"][g].copy())
                floats.append(obs["floats"][g].copy())
                masks.append(obs["mask"][g].copy())
                offered.append([c["name"].rstrip("+") for c in json.loads(env.envs.observe(int(g)))["screen"]["cards"]])
        for g in np.flatnonzero(env.step(acts)):
            done[g] = True
    return np.array(ints), np.array(floats), np.array(masks), offered


def log_probs(net, device, ints, floats, masks, batch=1024):
    out = []
    with torch.no_grad():
        for i in range(0, len(ints), batch):
            lg, _ = net(torch.from_numpy(ints[i:i + batch]).to(device).long(), torch.from_numpy(floats[i:i + batch]).to(device))
            lg = masked(lg.float(), torch.from_numpy(masks[i:i + batch]).to(device))
            out.append(torch.log_softmax(lg, -1).cpu().numpy())
    return np.concatenate(out)


def first_empty(block):
    """Each row's first empty slot (id 0) in a block of ids, or -1 when it is full."""
    empty = block == 0
    return np.where(empty.any(1), empty.argmax(1), -1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("network", help="a release weights file (.safetensors) or a run checkpoint (.pt)")
    p.add_argument("--runs", type=int, default=160, help="greedy runs per ascension")
    p.add_argument("--ascensions", default="10,20")
    p.add_argument("--min-positions", type=int, default=12)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--show", default="", help='pairs to print, "X>Y" separated by commas (X as printed: "card: ..." or "relic: ...")')
    p.add_argument("--out", help="the whole table as JSON: {\"X | Y\": [change, positions]}")
    a = p.parse_args()
    device = pick_device()
    net, character = load_network(a.network, device)
    layout = sts.layout()
    card_names, relic_ids = layout["card_names"], layout["relic_java_ids"]
    card_color = dict(re.findall(r'name: "([^"]+)", cost: -?\d+, card_type: CardType::\w+, color: CardColor::(\w+)',
                                 open(os.path.join(ROOT, "crates/sts-core/src/cards/data.rs")).read()))
    relic_info = {j: (n, c) for j, n, c in re.findall(r'java_id: "([^"]+)", name: "([^"]+)", tier: RelicTier::\w+, class: RelicClass::(\w+)',
                                                       open(os.path.join(ROOT, "crates/sts-core/src/relics/data.rs")).read())}
    own = COLOR[character]

    parts = [positions(net, device, a.runs, int(asc), character) for asc in a.ascensions.split(",")]
    ints = np.concatenate([q[0] for q in parts])
    floats = np.concatenate([q[1] for q in parts])
    masks = np.concatenate([q[2] for q in parts])
    offered = [o for q in parts for o in q[3]]
    print(f"{a.network}: {character}, {len(offered)} card-reward positions from {a.runs} runs at A{a.ascensions}", flush=True)
    base = log_probs(net, device, ints, floats, masks)

    d0, d1 = RUN_INT_AT["deck"]
    r0, r1 = RUN_INT_AT["run_relic"]
    rc0, _ = RUN_FLOAT_AT["run_relic_counter"]
    s0, _ = RUN_FLOAT_AT["run_scalars"]
    deck_slot, relic_slot, rows = first_empty(ints[:, d0:d1]), first_empty(ints[:, r0:r1]), np.arange(len(ints))
    cards = [i for i, n in enumerate(card_names) if card_color.get(n) in (own, "Colorless") and n not in ("Strike", "Defend")]
    relics = [i for i, j in enumerate(relic_ids) if j in relic_info and relic_info[j][1] in ("Shared", own)]
    where = {}
    for pos, names in enumerate(offered):
        for k, y in enumerate(names):
            where.setdefault(y, []).append((pos, PICK + k))

    table = {}

    def measure(x, new_ints, new_floats, ok):
        lp = log_probs(net, device, new_ints, new_floats, masks)
        for y, spots in where.items():
            spots = [s for s in spots if ok[s[0]]]
            if len(spots) >= a.min_positions:
                pos, act = np.array([s[0] for s in spots]), np.array([s[1] for s in spots])
                table[(x, y)] = (float((lp[pos, act] - base[pos, act]).mean()), len(spots))

    for cid in cards:
        ok = deck_slot >= 0
        new_ints, new_floats = ints.copy(), floats.copy()
        new_ints[rows[ok], d0 + deck_slot[ok]] = cid + 1
        new_floats[:, s0 + DECK_SIZE] += 1
        measure(f"card: {card_names[cid]}", new_ints, new_floats, ok)
    for rid in relics:
        ok = (relic_slot >= 0) & ~(ints[:, r0:r1] == rid + 1).any(1)
        new_ints, new_floats = ints.copy(), floats.copy()
        new_ints[rows[ok], r0 + relic_slot[ok]] = rid + 1
        new_floats[rows[ok], rc0 + relic_slot[ok]] = -1
        measure(f"relic: {relic_info[relic_ids[rid]][0]}", new_ints, new_floats, ok)

    items = sorted(table.items(), key=lambda kv: kv[1][0])
    fmt = lambda kv: f"  {kv[0][0]:28} -> {kv[0][1]:20} {kv[1][0]:+.2f}  ({kv[1][1]} positions)"  # noqa: E731
    print(f"{len(cards)} cards and {len(relics)} relics tried; {len(table)} pairs; typical change (median |change|) "
          f"{statistics.median(abs(v[0]) for v in table.values()):.2f}; pairs over 1.0: {sum(abs(v[0]) > 1 for v in table.values())}")
    print("\nThe strongest synergies (the change in the log-probability of the pick):")
    print("\n".join(fmt(kv) for kv in items[::-1][:a.top]))
    print("\nThe strongest anti-synergies:")
    print("\n".join(fmt(kv) for kv in items[:a.top]))
    if a.show:
        print("\nAsked for:")
        for pair in a.show.split(","):
            x, _, y = pair.partition(">")
            v = table.get((x.strip(), y.strip()))
            print(f"  {x.strip():28} -> {y.strip():20} " + (f"{v[0]:+.2f}  ({v[1]} positions)" if v else "not measured (offered too rarely, or no such item)"))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({f"{x} | {y}": v for (x, y), v in table.items()}, f, indent=0)


if __name__ == "__main__":
    main()
