"""Which fights does a checkpoint lose, and could they be won?

    .venv/bin/python python/analysis/loss_report.py python/runs/ironclad-fights-2/latest.pt [fights] [retries]

Plays the fixed random-deck fight set of baselines.py greedily, then replays every lost fight
`retries` times with sampled moves (each fight is fixed by its seed: deck, encounter and shuffles
are the same, only the moves differ). A loss that some replay wins was winnable, so
"greedy + replays" is a rough lower bound on the best possible win rate.

Prints win rates per encounter (greedy, scripted, with replays) and writes them to
<run>/loss_report.json.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

from analysis.baselines import ACT_OF, scripted_policy
from core.fight_model import FightNet, masked, pick_device, to_tensors
from core.sts_env import FightEnv

EVAL_SEED = 100_000
BATCH = 128


def play_set(policy, fights):
    """The first `fights` fights of the eval set (same seeds and batch as baselines.evaluate)."""
    env = FightEnv(BATCH, seed=EVAL_SEED)
    rng = np.random.default_rng(EVAL_SEED)
    per_game = -(-fights // BATCH)
    done = np.zeros(BATCH, dtype=int)
    out = []
    while (done < per_game).any():
        status = env.step(policy(env.observe(), rng))
        for g in np.flatnonzero(status):
            if done[g] < per_game:
                out.append(json.loads(env.last_episode(int(g))))
            done[g] += 1
    return out[:fights]


def retry(net, device, seeds, tries):
    """Replays each seed's fight `tries` times with sampled moves; returns wins per seed."""
    envs = [FightEnv(1, seed=s) for s in seeds for _ in range(tries)]
    owner = [s for s in seeds for _ in range(tries)]
    live = list(range(len(envs)))
    wins = defaultdict(int)
    while live:
        obs = [envs[i].observe() for i in live]
        batch = {k: np.concatenate([o[k] for o in obs]) for k in ("ints", "floats", "mask")}
        ints, floats, mask = to_tensors(batch, device)
        with torch.no_grad():
            logits, _ = net(ints, floats)
            acts = torch.distributions.Categorical(logits=masked(logits.float(), mask)).sample().cpu().numpy()
        still = []
        for i, a in zip(live, acts):
            status = envs[i].step([a])[0]
            if status == 1:
                wins[owner[i]] += 1
            if status == 0:
                still.append(i)
        live = still
    return wins


def main():
    ckpt = sys.argv[1]
    fights = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    tries = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    device = pick_device()
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    net = FightNet(d=a["d"], layers=a["layers"], heads=a["heads"]).to(device).eval()
    net.load_state_dict(ck["net"])

    @torch.no_grad()
    def greedy(obs, rng):
        ints, floats, mask = to_tensors(obs, device)
        return masked(net(ints, floats)[0], mask).argmax(-1).cpu().numpy()

    eps = play_set(greedy, fights)
    scripted = play_set(scripted_policy, fights)
    lost = [e["seed"] for e in eps if not e["won"]]
    print(f"{ckpt}: greedy wins {np.mean([e['won'] for e in eps]):.1%} of {len(eps)}; replaying {len(lost)} losses x {tries}")
    wins = retry(net, device, lost, tries)

    rows = defaultdict(lambda: {"n": 0, "greedy": 0, "scripted": 0, "replays": 0, "hp_lost_won": []})
    scripted_by_seed = {s["seed"]: s for s in scripted}
    for e in eps:
        s = scripted_by_seed[e["seed"]]
        r = rows[e["encounter"]]
        r["n"] += 1
        r["greedy"] += e["won"]
        r["scripted"] += s["won"]
        r["replays"] += e["won"] or wins.get(e["seed"], 0) > 0
        if e["won"]:
            r["hp_lost_won"].append(e["max_hp"] - e["hp"])

    order = sorted(rows, key=lambda k: (ACT_OF.get(k, 9), rows[k]["greedy"] / rows[k]["n"]))
    print(f"\n{'encounter':34} act  fights  scripted  greedy  +replays  lost  winnable  hp lost (wins)")
    report = {}
    for act in (1, 2, 3, 4):
        names = [k for k in order if ACT_OF.get(k) == act]
        if not names:
            continue
        for k in names:
            r = rows[k]
            n = r["n"]
            losses = n - r["greedy"]
            saved = r["replays"] - r["greedy"]
            hp = np.mean(r["hp_lost_won"]) if r["hp_lost_won"] else float("nan")
            print(f"{k:34} {act:3}  {n:6}  {r['scripted'] / n:8.0%}  {r['greedy'] / n:6.0%}  {r['replays'] / n:8.0%}  "
                  f"{losses:4}  {saved:8}  {hp:6.1f}")
            report[k] = {"act": act, "fights": n, "scripted": r["scripted"] / n, "greedy": r["greedy"] / n,
                         "with_replays": r["replays"] / n, "hp_lost_when_won": hp}
        tot = {f: sum(rows[k][f] for k in names) for f in ("n", "greedy", "scripted", "replays")}
        print(f"{'  Act ' + str(act) + ' total':34} {'':3}  {tot['n']:6}  {tot['scripted'] / tot['n']:8.0%}  "
              f"{tot['greedy'] / tot['n']:6.0%}  {tot['replays'] / tot['n']:8.0%}\n")
    tot = {f: sum(r[f] for r in rows.values()) for f in ("n", "greedy", "scripted", "replays")}
    print(f"all: scripted {tot['scripted'] / tot['n']:.1%}  greedy {tot['greedy'] / tot['n']:.1%}  "
          f"greedy + {tries} replays {tot['replays'] / tot['n']:.1%}")
    with open(os.path.join(os.path.dirname(ckpt), "loss_report.json"), "w") as f:
        json.dump({"checkpoint": ckpt, "tries": tries, "encounters": report}, f, indent=2)


if __name__ == "__main__":
    main()
