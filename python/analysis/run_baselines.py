"""Whole-run baselines: the trained fight network in combat, simple players everywhere else.

    .venv/bin/python python/analysis/run_baselines.py <fight checkpoint> [runs]

Two players for the screens between fights:
- random: a uniformly random legal choice (potion discards excluded);
- scripted: Neow's first blessing without a drawback; on the map, a rest site when below half
  HP, otherwise a fight or event, elites only above 70% HP; takes every reward (cards: the
  first offered while the deck has under 25 cards); rests below 60% HP, otherwise upgrades;
  the first boss relic; at shops removes a card when it can, then leaves; random event
  options and deck cards.

Both play the same fixed seeds (from 300,000) at ascension 0 with the final act on. Scores:
floor reached, how far runs get (Act 1-4, Heart beaten), and where they die. Results go to
python/results/run_baselines.json.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

from core.fight_model import FightNet, masked, pick_device
from core.sts_env import FLOAT_LEN, INT_LEN, LAYOUT, RunEnv

A = LAYOUT["actions"]
PICK, MAX_OPTIONS = A["pick"]
CONFIRM = A["confirm"]
POTION, _, TARGETS = A["potion"]
SCREENS = ["-"] + LAYOUT["screens"]
KINDS = ["-"] + LAYOUT["option_kinds"]
COMBAT = SCREENS.index("combat")
ROOMS = ["-", "monster", "elite", "event", "rest", "shop", "treasure", "boss"]
EVAL_SEED = 300_000


def non_potion(mask):
    """Legal slots, leaving out drinking and discarding potions outside combat."""
    m = mask.copy()
    m[POTION:POTION + 5 * TARGETS] = False
    return m if m.any() else mask


def random_choices(obs, g, rng):
    return rng.choice(np.flatnonzero(non_potion(obs["mask"][g])))


def scripted_choices(obs, g, rng):
    m = non_potion(obs["mask"][g])
    picks = [k for k in range(MAX_OPTIONS) if m[PICK + k]]
    kinds = [KINDS[k] for k in obs["i_opt_kind"][g]]
    sc = obs["f_run_scalars"][g]
    hp_frac = sc[0] / max(sc[1], 1)
    deck = sc[6]
    screen = SCREENS[obs["i_screen"][g, 0]]
    if screen == "neow":
        good = [k for k in picks if obs["i_opt_drawback"][g, k] == 1]
        return PICK + (good or picks)[0]
    if screen == "map":
        def score(k):
            room = ROOMS[obs["i_opt_room"][g, k]]
            if room == "rest":
                return 3 if hp_frac < 0.5 else 1
            if room == "elite":
                return 2 if hp_frac > 0.7 else -2
            return {"monster": 2, "event": 2, "shop": 1, "treasure": 2}.get(room, 0)
        return PICK + max(picks, key=lambda k: (score(k), -k))
    if screen == "rewards":
        # (Skipping a card reward leaves it on the list, so don't open it again.)
        take = [k for k in picks if kinds[k] != "sapphire_key" and not (kinds[k] == "cards" and deck >= 25)]
        return PICK + take[0] if take else CONFIRM
    if screen == "card_reward":
        cards = [k for k in picks if kinds[k] == "pick_card"]
        return PICK + cards[0] if cards and deck < 25 else (CONFIRM if m[CONFIRM] else PICK + picks[0])
    if screen == "rest":
        if hp_frac < 0.6 and "rest" in kinds:
            return PICK + kinds.index("rest")
        if "smith" in kinds:
            return PICK + kinds.index("smith")
        return PICK + picks[0] if picks else CONFIRM
    if screen == "boss_relic":
        return PICK + picks[0] if picks else CONFIRM
    if screen == "shop":
        purge = [k for k in picks if kinds[k] == "shop_purge"]
        return PICK + purge[0] if purge else CONFIRM
    return rng.choice(np.flatnonzero(m))


def first_episodes(out, n):
    """The first `n` of `out` ((episode number in its game, game), summary) in game order, never in
    the order they ended (that keeps the shortest runs; train_runs.first_episodes)."""
    return [e for _, e in sorted(out, key=lambda x: x[0])[:n]]


def evaluate(choices, fight_net, device, runs, batch=256, seed=EVAL_SEED, **env_kwargs):
    """Plays `runs` runs (the first episodes of each game) and returns their summaries."""
    batch = min(batch, runs)
    env = RunEnv(batch, seed=seed, **env_kwargs)
    rng = np.random.default_rng(seed)
    per_game = -(-runs // batch)
    done = np.zeros(batch, dtype=int)
    out = []
    while (done < per_game).any():
        obs = env.observe()
        acts = np.zeros(batch, dtype=np.int64)
        combat = obs["i_screen"][:, 0] == COMBAT
        if combat.any():
            ints = torch.from_numpy(obs["ints"][combat, :INT_LEN].astype(np.int64)).to(device)
            floats = torch.from_numpy(obs["floats"][combat, :FLOAT_LEN]).to(device)
            mask = torch.from_numpy(obs["mask"][combat]).to(device)
            with torch.no_grad():
                acts[combat] = masked(fight_net(ints, floats)[0], mask).argmax(-1).cpu().numpy()
        for g in np.flatnonzero(~combat):
            acts[g] = choices(obs, g, rng)
        status = env.step(acts)
        for g in np.flatnonzero(status):
            if done[g] < per_game:
                out.append(((done[g], g), json.loads(env.last_episode(int(g)))))
            done[g] += 1
    return first_episodes(out, runs)


def summarize(results):
    floors = np.array([r["floor"] for r in results])
    acts = Counter(r["act"] for r in results)
    won = np.array([r["won"] for r in results])
    return {
        "runs": len(results),
        "win_rate": float(won.mean()),
        "floor": float(floors.mean()),
        "floor_median": float(np.median(floors)),
        "reached_act": {str(a): float(np.mean([r["act"] >= a for r in results])) for a in (2, 3, 4)},
        "died_in_act": {str(a): acts.get(a, 0) / len(results) for a in (1, 2, 3, 4)},
        "timed_out": int(sum(r.get("timed_out", False) for r in results)),
    }


def load_fight_net(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    net = FightNet(d=a["d"], layers=a["layers"], heads=a["heads"]).to(device).eval()
    net.load_state_dict(ck["net"])
    return net


def main():
    ckpt = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    device = pick_device()
    net = load_fight_net(ckpt, device)
    report = {"fight_checkpoint": ckpt}
    for name, choices in (("random", random_choices), ("scripted", scripted_choices)):
        t = time.time()
        r = summarize(evaluate(choices, net, device, runs))
        r["seconds"] = round(time.time() - t, 1)
        report[name] = r
        reach = "  ".join(f"A{k} {v:.0%}" for k, v in r["reached_act"].items())
        print(f"{name:9} floor {r['floor']:5.1f} (median {r['floor_median']:.0f})  reached [{reach}]  won {r['win_rate']:.1%}  ({r['seconds']}s)")
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "run_baselines.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
