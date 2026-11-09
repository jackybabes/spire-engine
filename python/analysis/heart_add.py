"""What would the network's own decks need to beat the Heart? The decks its runs brought there (search_eval.py --free-keys --decks --dump: runs that got
past the Spire elite) and, for comparison, her winners' (human_bosses.py lines), each played through the Heart greedily from the same share of HP, with no potions, on several seeds:
the win rate as they are and, for each card or set of cards in --add, the same fights with those added. So it asks of each card what fights_by_card.py's removal asks
of her decks, on the decks the network actually makes: the difference, by deck and then averaged, with the standard error over decks.

    .venv/bin/python python/analysis/heart_add.py python/runs/silent-runs-10/update_700.pt --dumps ~/scripts/dumps/r10-700-keys,~/scripts/dumps/r12-100 --gpu-gb 3.5 --out heart_add.json

The checkpoint's fight network plays, as in fights_by_card.py. An added card comes on top of the deck: nothing goes to make room for it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import collections
import glob
import json
import time

import numpy as np
import torch

from analysis.archetypes import arch
from analysis.fights_by_card import SCENARIOS, LAYOUT, evenly, mean_se, play_fights
from core.fight_model import pick_device
from training.train_runs import SCENARIO_EVAL_SEED, load, load_fight_net, read_scenarios

ADD = ("Wraith Form+;Piercing Wail+;Catalyst+;Footwork+;Well-Laid Plans+;Acrobatics+;Tactician+;Grand Finale+;Apparition+;Noxious Fumes+;Deadly Poison+;Bouncing Flask+;After Image+;Malaise+;"
       "Adrenaline+;Burst+;Crippling Cloud+;Corpse Explosion+;Backflip+;Dodge and Roll+;Blur+;Leg Sweep+;Accuracy+;Infinite Blades+;Blade Dance+;Defend_G;Strike_G;"
       "Catalyst+,Noxious Fumes+,Deadly Poison+;Wraith Form+,Piercing Wail+;Wraith Form+,Piercing Wail+,Footwork+;Acrobatics+,Tactician+,Grand Finale+")
HEART = {"character": "THE_SILENT", "ascension": 20, "act": 4, "floor": 56, "bosses": ["The Heart"], "potions": []}
MAX_RELICS = 30
# Several cards are called Strike and Defend: the Silent's.
JAVA = {**dict(zip(LAYOUT["card_names"], LAYOUT["card_java_ids"])), "Strike": "Strike_G", "Defend": "Defend_G", "Strike_G": "Strike_G", "Defend_G": "Defend_G"}


def java_card(name):
    """A card's [game id, upgrades] from its name, "+" for upgraded."""
    base = name.rstrip("+")
    if base not in JAVA:
        sys.exit(f"unknown card: {name}")
    return [JAVA[base], 1 if name.endswith("+") else 0]


def own_decks(prefixes):
    """The decks and relics of the runs in search_eval dumps that reached the Heart: [{"deck": [[game id, upgrades]...], "relics": [...], "max_hp": n, "won": bool}...]."""
    decks = []
    for prefix in prefixes:
        paths = glob.glob(os.path.expanduser(prefix) + ".*.jsonl")
        if not paths:
            sys.exit(f"no dump: {prefix}")
        for path in paths:
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if r["floor"] >= 56 and r["act"] >= 4 and "deck" in r and "relics" in r:
                        decks.append({"deck": [java_card(c) for c in r["deck"]], "relics": r["relics"], "max_hp": r["max_hp"], "won": r["won"]})
    return decks


def her_decks(paths):
    """Her winners' decks at the Heart, as in human_bosses.py's lines, and their mean share of HP there."""
    decks, shares = [], []
    for path in paths.split(","):
        for line in read_scenarios(os.path.expanduser(path)):
            r = json.loads(line)
            if r["act"] == 4 and r["bosses"] == ["The Heart"] and len(r["relics"]) < MAX_RELICS:
                decks.append({"deck": r["deck"], "relics": r["relics"], "max_hp": r["max_hp"], "won": True})
                shares.append(r["hp"] / max(r["max_hp"], 1))
    return decks, float(np.mean(shares))


def fights_of(decks, seeds, share, added=()):
    """Each deck on each of `seeds` seeds, the cards `added`: the same seeds whatever is added, so that the fights pair up."""
    return [dict(HEART, deck=d["deck"] + [list(c) for c in added], relics=d["relics"], max_hp=d["max_hp"], hp=max(int(d["max_hp"] * share), 1), seed=SCENARIO_EVAL_SEED + i * seeds + s)
            for i, d in enumerate(decks) for s in range(seeds)]


def usable(env, decks):
    """The decks the engine can set up (a relic it can't place, say, leaves one out)."""
    keep = []
    for d in decks:
        try:
            if env.set_scenarios([json.dumps(fights_of([d], 1, 0.5)[0])], 1.0, True) == 0:
                keep.append(d)
        except ValueError:
            pass
    return keep


def by_deck(eps, seeds):
    """Each deck's share of its fights won."""
    won = np.array([e["won"] for e in eps], dtype=float)
    return won.reshape(-1, seeds).mean(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--dumps", required=True, help="search_eval dumps' prefixes, comma-separated")
    p.add_argument("--human", default=SCENARIOS, help="her lines, comma-separated")
    p.add_argument("--human-decks", type=int, default=400, help="her winners' decks, evenly through them")
    p.add_argument("--add", default=ADD, help="what to add, each set's cards comma-separated, the sets semicolon-separated")
    p.add_argument("--seeds", type=int, default=8, help="fights a deck")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--gpu-gb", type=float, default=3.5, help="GPU memory cap, GB (PyTorch's own, without the CUDA context's ~0.45 GB; 0 = none)")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="everything as JSON (written after each set)")
    args = p.parse_args()
    device = pick_device(args.device)
    if device.type == "cuda" and args.gpu_gb:
        index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_per_process_memory_fraction(min(args.gpu_gb * 2 ** 30 / torch.cuda.get_device_properties(index).total_memory, 1.0), index)

    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    net, ck = load(args.checkpoint, device)
    net.load_state_dict(ck["net"])
    fight_path = ck["args"].get("fight_net")
    net = load_fight_net(fight_path, device) if fight_path else net.eval()
    character = ck["args"].get("character", "ironclad")
    import sts
    env = sts.Envs(1, mode="run", practice=1.0, character=character)
    hers, share = her_decks(args.human)
    hers = usable(env, evenly(hers, args.human_decks))
    mine = usable(env, own_decks(args.dumps.split(",")))
    print(f"{args.checkpoint} (update {ck.get('update')}), fights by {fight_path or 'itself'}: {len(mine)} of its runs' decks at the Heart "
          f"({sum(d['won'] for d in mine)} won there; {dict(collections.Counter(arch(c[0] for c in d['deck']) for d in mine))}), {len(hers)} of her winners', {args.seeds} fights a deck, each from {share:.0%} of its HP, no potions", flush=True)

    def play(decks, added=()):
        return by_deck(play_fights(net, device, amp, fights_of(decks, args.seeds, share, added), args.batch, character), args.seeds)

    groups = {"own": mine, "hers": hers}
    base = {k: play(d) for k, d in groups.items()}
    result = {"checkpoint": args.checkpoint, "update": ck.get("update"), "decks": {k: len(d) for k, d in groups.items()}, "seeds": args.seeds, "share": share,
              "base": {k: list(mean_se(v)) for k, v in base.items()}, "added": {}}
    print("won as they are: " + ", ".join(f"{k} {100 * mean_se(v)[0]:.1f} ±{100 * mean_se(v)[1]:.1f}" for k, v in base.items()), flush=True)
    print("each set added (with minus as they are, in points, ± its standard error over decks): " + ", ".join(groups), flush=True)
    for names in args.add.split(";"):
        added = [java_card(c) for c in names.split(",")]
        t = time.time()
        row = {}
        for k, d in groups.items():
            diff = play(d, added) - base[k]
            row[k] = list(mean_se(diff))
        result["added"][names] = row
        print(f"{names:48} " + "  ".join(f"{k} {100 * v[0]:+5.1f} ±{100 * v[1]:3.1f}" for k, v in row.items()) + f"  ({time.time() - t:.0f} s)", flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f)


if __name__ == "__main__":
    main()
