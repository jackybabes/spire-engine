"""What a run checkpoint chooses between fights, on the 512 fixed eval runs (greedy).

    .venv/bin/python python/analysis/run_choices.py <run checkpoint>
    .venv/bin/python python/analysis/run_choices.py <run checkpoint> --force-upgrade 0.85 0.7 0.5

Prints rest-site choices by HP and by act, map rooms taken against offered, card rewards,
reward screens, shops, boss relics and final deck sizes. With --force-upgrade, it instead
replays the same runs once per HP threshold, upgrading at rest sites whenever HP is at least
that fraction, and compares the win rate with the network's own choices (ironclad-runs-3 at update 600:
71.1% on its own, 76.4% when made to upgrade above 70% HP).
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

SCREENS = ["-"] + LAYOUT["screens"]
KINDS = ["-"] + LAYOUT["option_kinds"]
ROOMS = ["-", "monster", "elite", "event", "rest", "shop", "treasure", "boss"]
PICK, _ = LAYOUT["actions"]["pick"]
CONFIRM = LAYOUT["actions"]["confirm"]
N = 512
# The runs' character and ascension: the checkpoint run's, or --ascension (main sets them; they were the
# Ironclad at A0 for every checkpoint until 2026-10-28, when a Watcher network's report came out at floor 12).
RUN = {"character": "ironclad", "ascension": 0}


def play(net, device, on_step=None, adjust=None):
    """Plays the fixed runs greedily; `on_step(obs, actions, live, env)` sees each step,
    `adjust(obs, actions, live, env)` may change the actions. Returns the run summaries."""
    env = RunEnv(N, seed=EVAL_SEED, max_steps=8000, character=RUN["character"], ascension=RUN["ascension"])
    res = [None] * N
    while any(r is None for r in res):
        obs = env.observe()
        with torch.no_grad():
            lg, _ = net(torch.from_numpy(obs["ints"]).to(device).long(), torch.from_numpy(obs["floats"]).to(device))
        acts = masked(lg.float(), torch.from_numpy(obs["mask"]).to(device)).argmax(-1).cpu().numpy()
        live = np.array([r is None for r in res])
        if adjust:
            adjust(obs, acts, live, env)
        if on_step:
            on_step(obs, acts, live, env)
        for g in np.flatnonzero(env.step(acts)):
            if res[g] is None:
                res[g] = json.loads(env.last_episode(int(g)))
    return res


def describe(res):
    return (f"won {np.mean([r['won'] for r in res]):.1%}, floor {np.mean([r['floor'] for r in res]):.1f}, "
            f"reached Act 3 {np.mean([r['act'] >= 3 for r in res]):.0%}")


def choices(net, device):
    rest = []  # (hp fraction, act, choice)
    shop, card, path, offered, reward, boss = (collections.Counter() for _ in range(6))
    deck_end = [0] * N

    def kind(obs, g, a):
        if PICK <= a < CONFIRM:
            return KINDS[obs["i_opt_kind"][g, a - PICK]]
        return "proceed/skip" if a == CONFIRM else "potion"

    def on_step(obs, acts, live, env):
        for g in np.flatnonzero(live):
            sc = obs["f_run_scalars"][g]
            screen = SCREENS[obs["i_screen"][g, 0]]
            deck_end[g] = sc[6]
            k = kind(obs, g, acts[g])
            if screen == "rest":
                rest.append((sc[0] / max(sc[1], 1), int(sc[4]), k))
            elif screen == "shop":
                shop[k] += 1
            elif screen == "card_reward":
                card[k] += 1
            elif screen == "rewards":
                reward[k] += 1
            elif screen == "boss_relic":
                boss[k] += 1
            elif screen == "map":
                path[ROOMS[obs["i_opt_room"][g, acts[g] - PICK]] if PICK <= acts[g] < CONFIRM else "?"] += 1
                for j in range(64):
                    if obs["mask"][g, PICK + j]:
                        offered[ROOMS[obs["i_opt_room"][g, j]]] += 1

    res = play(net, device, on_step=on_step)
    print(describe(res))
    print("REST SITES", dict(collections.Counter(c for _, _, c in rest)))
    for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)):
        sub = [c for f, _, c in rest if lo <= f < hi]
        if sub:
            print(f"  HP {lo:.0%}-{min(hi, 1):.0%}: {len(sub):5} visits, rest {sub.count('rest') / len(sub):.0%}, smith {sub.count('smith') / len(sub):.0%}")
    for act in (1, 2, 3):
        sub = [c for _, a, c in rest if a == act]
        if sub:
            print(f"  act {act}: {len(sub):5} visits, rest {sub.count('rest') / len(sub):.0%}, smith {sub.count('smith') / len(sub):.0%}")
    print("MAP chosen", dict(path))
    print("    offered", dict(offered))
    print(f"  elite taken when offered {path['elite'] / max(offered['elite'], 1):.0%}, rest taken when offered {path['rest'] / max(offered['rest'], 1):.0%}")
    print("CARD REWARDS", dict(card))
    print("REWARD SCREEN", dict(reward))
    print("SHOP", dict(shop))
    print("BOSS RELIC", dict(boss))
    won = [d for d, r in zip(deck_end, res) if r["won"]]
    lost = [d for d, r in zip(deck_end, res) if not r["won"]]
    print(f"DECK SIZE at the end: wins {np.mean(won):.1f}, losses {np.mean(lost):.1f}")


def force_upgrade(net, device, thresholds):
    smith = KINDS.index("smith")
    rest = SCREENS.index("rest")
    print("network's own choices:", describe(play(net, device)), flush=True)
    for threshold in thresholds:
        changed = [0]

        def adjust(obs, acts, live, env):
            for g in np.flatnonzero((obs["i_screen"][:, 0] == rest) & live):
                sc = obs["f_run_scalars"][g]
                k = np.flatnonzero(obs["i_opt_kind"][g] == smith)
                if sc[0] / max(sc[1], 1) >= threshold and k.size and obs["mask"][g, PICK + k[0]]:
                    changed[0] += acts[g] != PICK + k[0]
                    acts[g] = PICK + k[0]

        res = play(net, device, adjust=adjust)
        print(f"upgrade when HP >= {threshold:.0%}: {describe(res)} ({changed[0]} rests changed to upgrades)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--force-upgrade", type=float, nargs="+", metavar="HP_FRACTION")
    p.add_argument("--ascension", type=int, default=None, help="the runs' ascension (default: the checkpoint run's --ascension)")
    args = p.parse_args()
    device = pick_device()
    net, ck = load(args.checkpoint, device)
    net.load_state_dict(ck["net"])
    net.eval()
    RUN["character"] = ck["args"].get("character", "ironclad")
    RUN["ascension"] = args.ascension if args.ascension is not None else ck["args"].get("ascension", 0)
    print(f"{args.checkpoint} (update {ck['update']}), {RUN['character']} at A{RUN['ascension']}")
    if args.force_upgrade:
        force_upgrade(net, device, args.force_upgrade)
    else:
        choices(net, device)


if __name__ == "__main__":
    main()
