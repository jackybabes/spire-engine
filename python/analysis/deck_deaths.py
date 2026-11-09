"""Where a deck network's runs die, and in what shape (jack, 2026-10-30: "yes run the analysis on where runs die").

    .venv/bin/python python/analysis/deck_deaths.py python/runs/ironclad-deck-2/update_300.pt [--runs 512]

Plays --runs A20 runs greedily on the fixed eval seeds as train_runs.evaluate does: the checkpoint choosing between fights,
its --fight-net fighting, the lookup at card rewards if it trained with one. For each run that dies, the fight it died
in (act, floor, room: hallway, elite or boss; the encounter) and the run going into it (its HP share, deck size, gold,
potions held, floor); reports deaths by act and room, the encounters that end most runs, and those numbers for the runs
that die in each act against the ones that get further.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import contextlib
import json
from collections import Counter, defaultdict

import numpy as np
import torch

from core.fight_model import masked
from core.lookup import Lookup
from core.run_model import COMBAT, RUN_FLOAT_AT, RUN_INT_AT
from core.split import SplitPolicy
from core.sts_env import RunEnv
from training.train_runs import EVAL_SEED, load, load_fight_net

BOSSES = {"The Guardian", "Hexaghost", "Slime Boss", "Automaton", "Collector", "Champ", "Awakened One", "Time Eater",
          "Donu and Deca", "The Heart"}
ELITES = {"Gremlin Nob", "Lagavulin", "3 Sentries", "Gremlin Leader", "Slavers", "Book of Stabbing", "Giant Head", "Nemesis",
          "Reptomancer", "Shield and Spear"}
SCREEN_AT = RUN_INT_AT["screen"][0]
RS = RUN_FLOAT_AT["run_scalars"][0]  # hp, max hp, gold, floor, act, ascension, deck size, ...
POTIONS = RUN_INT_AT["run_potion"]


def room_of(enc):
    return "boss" if enc in BOSSES else "elite" if enc in ELITES else "hallway"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--runs", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--first", type=int, default=0, help="the first eval seed's index (EVAL_SEED + it): more runs on new seeds")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else contextlib.nullcontext
    net, ck = load(a.checkpoint, device)
    net.load_state_dict(ck["net"])
    net.eval()
    args = ck["args"]
    fight_net = load_fight_net(args["fight_net"], device) if args.get("fight_net") else None
    policy = SplitPolicy(fight_net, net) if fight_net is not None else net
    lookup = (Lookup(fight_net, device, amp, draws=args.get("lookup_draws", 4), every=args.get("lookup_every", False))
              if args.get("lookup") and fight_net is not None else None)
    print(f"{a.checkpoint} (update {ck.get('update')}): {a.runs} A20 runs, fights by {args.get('fight_net')}, lookup {bool(lookup)}", flush=True)

    out = []
    for start in range(a.first, a.first + a.runs, a.batch):
        n = min(a.batch, a.first + a.runs - start)
        env = RunEnv(n, seed=EVAL_SEED + start, ascension=args.get("ascension", 20), max_steps=8000, character=args.get("character", "ironclad"))
        res = [None] * n
        before = [None] * n  # each game's state entering its current fight
        was_combat = np.zeros(n, dtype=bool)
        last_run = [None] * n  # its state at its last row out of combat
        while any(r is None for r in res):
            obs = env.observe()
            if lookup is not None:
                obs = lookup.fill(env.envs, obs)
            ints, floats = obs["ints"], obs["floats"]
            combat = ints[:, SCREEN_AT] == COMBAT
            for g in np.flatnonzero(~combat):
                last_run[g] = (floats[g, RS:RS + 7].copy(), int((ints[g, POTIONS[0]:POTIONS[1]] > 0).sum()))
            for g in np.flatnonzero(combat & ~was_combat):
                if last_run[g] is not None:
                    sc, pots = last_run[g]
                    before[g] = {"hp_share": sc[0] / max(sc[1], 1), "gold": sc[2], "floor": sc[3], "deck": sc[6], "potions": pots}
            was_combat = combat
            with torch.no_grad(), amp():
                logits, _ = policy(torch.from_numpy(ints).to(device).long(), torch.from_numpy(floats).to(device))
            acts = masked(logits.float(), torch.from_numpy(obs["mask"]).to(device)).argmax(-1).cpu().numpy()
            for g in np.flatnonzero(env.step(acts)):
                if res[g] is None:
                    e = json.loads(env.last_episode(int(g)))
                    e["entering"] = before[g]
                    res[g] = e
                was_combat[g] = False
                before[g] = last_run[g] = None
        out += res
        print(f"  {start + n - a.first} of {a.runs} runs", flush=True)

    died = [e for e in out if not e["won"] and e.get("died_in")]
    print(f"\n{len(out)} runs: {len(died)} died in a fight, {sum(e['won'] for e in out)} won, "
          f"{sum(e.get('timed_out', False) for e in out)} timed out; floor {np.mean([e['floor'] for e in out]):.1f}")
    print("deaths by act and room (share of all runs):")
    for act in (1, 2, 3, 4):
        sub = [e for e in died if e["act"] == act]
        if sub:
            rooms = Counter(room_of(e["died_in"]) for e in sub)
            print(f"  Act {act}: {len(sub):4} ({100 * len(sub) / len(out):4.1f}%)  " +
                  ", ".join(f"{r} {rooms[r]} ({100 * rooms[r] / len(out):.1f}%)" for r in ("hallway", "elite", "boss")))
    print("the encounters that end most runs:")
    for (enc, act), k in Counter((e["died_in"], e["act"]) for e in died).most_common(15):
        sub = [e for e in died if e["died_in"] == enc and e["act"] == act and e["entering"]]
        hp = np.mean([e["entering"]["hp_share"] for e in sub]) if sub else float("nan")
        print(f"  Act {act} {room_of(enc):7} {enc:24} {k:4} ({100 * k / len(out):4.1f}% of runs); entering at {100 * hp:3.0f}% HP")

    print("going into the fatal fight, by where the run died (and the runs that reached Act 3, at their last fight):")
    groups = defaultdict(list)
    for e in died:
        if e["entering"]:
            groups[f"died in Act {e['act']} {room_of(e['died_in'])}"].append(e["entering"])
    for e in out:
        if e["act"] >= 3 and e["entering"]:
            groups["reached Act 3"].append(e["entering"])
    for name in sorted(groups, key=lambda k: (k == "reached Act 3", k)):
        g = groups[name]
        print(f"  {name:26} {len(g):4} runs: HP {100 * np.mean([x['hp_share'] for x in g]):3.0f}%, deck {np.mean([x['deck'] for x in g]):4.1f} "
              f"cards, gold {np.mean([x['gold'] for x in g]):4.0f}, potions {np.mean([x['potions'] for x in g]):.1f}, "
              f"floor {np.mean([x['floor'] for x in g]):4.1f}")


if __name__ == "__main__":
    main()
