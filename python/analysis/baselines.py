"""Baseline fight players, scored on a fixed set of fights for the network to beat.

    .venv/bin/python python/analysis/baselines.py [fights]

Two players:
- random: a uniformly random legal action slot;
- scripted: block when the incoming damage beats the block, otherwise the hardest-hitting
  attack at the weakest monster, otherwise any other playable card, then end the turn; picks
  the first option in a card selection; never drinks potions.

Two fight sets, each a fixed list of seeds (so later players face the same fights):
- random decks against every encounter (what stage 4 trains on);
- the starter deck against the 20 Act 1 encounters.

Scores: win rate, HP lost per fight (of 80), and turns per fight. Results go to
python/results/baselines.json.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import json
import os
import sys
import time

import numpy as np

import sts
from core.sts_env import LAYOUT, FightEnv

A = LAYOUT["actions"]
PLAY, _, TARGETS = A["play"]
END_TURN = A["end_turn"]
PICK, _ = A["pick"]
CONFIRM = A["confirm"]

ACT1 = [
    "Cultist", "Jaw Worm", "2 Louse", "Small Slimes", "Blue Slaver", "Gremlin Gang", "Looter",
    "Large Slime", "Lots of Slimes", "Exordium Thugs", "Exordium Wildlife", "Red Slaver",
    "3 Louse", "2 Fungi Beasts", "Gremlin Nob", "Lagavulin", "3 Sentries", "The Guardian",
    "Hexaghost", "Slime Boss",
]


ACT_OF = {}
for _act, _names in enumerate(sts.encounters_by_act(), start=1):
    for _name in _names:
        ACT_OF.setdefault(_name, _act)


def random_policy(obs, rng):
    return np.array([rng.choice(np.flatnonzero(m)) for m in obs["mask"]])


def scripted_policy(obs, rng):
    mask = obs["mask"]
    n = mask.shape[0]
    hand = obs["f_hand"].reshape(n, 10, 7)          # cost, upgrades, damage, block, magic, playable, targeted
    mon = obs["f_monster"].reshape(n, 6, 8)          # hp, max hp, block, intent dmg, hits, targetable, ...
    block = obs["f_scalars"][:, 2]
    out = np.empty(n, dtype=np.int64)
    for g in range(n):
        m = mask[g]
        picks = np.flatnonzero(m[PICK:PICK + 64])
        if picks.size or m[CONFIRM]:
            out[g] = PICK + picks[0] if picks.size else CONFIRM
            continue
        incoming = float((mon[g, :, 3] * mon[g, :, 4]).sum())
        # The weakest targetable monster.
        alive = np.flatnonzero(mon[g, :, 5] > 0)
        target = 1 + alive[np.argmin(mon[g, alive, 0])] if alive.size else 0

        def slot(h):
            if hand[g, h, 6] > 0:
                s = PLAY + h * TARGETS + target
                if m[s]:
                    return s
                ok = np.flatnonzero(m[PLAY + h * TARGETS + 1:PLAY + (h + 1) * TARGETS])
                return PLAY + h * TARGETS + 1 + ok[0] if ok.size else None
            s = PLAY + h * TARGETS
            return s if m[s] else None

        playable = [h for h in range(10) if hand[g, h, 5] > 0 and slot(h) is not None]
        choice = None
        if incoming > block[g]:
            blockers = [h for h in playable if hand[g, h, 3] > 0]
            if blockers:
                choice = max(blockers, key=lambda h: hand[g, h, 3])
        if choice is None:
            attacks = [h for h in playable if hand[g, h, 2] > 0]
            if attacks:
                choice = max(attacks, key=lambda h: hand[g, h, 2])
        if choice is None and playable:
            choice = playable[0]
        out[g] = slot(choice) if choice is not None else END_TURN
    return out


def evaluate(policy, fights, seed, batch=128, **env_kwargs):
    """Plays `fights` fights (the first episodes of each game, all from `seed`) with `policy`."""
    env = FightEnv(batch, seed=seed, **env_kwargs)
    rng = np.random.default_rng(seed)
    per_game = -(-fights // batch)
    done = np.zeros(batch, dtype=int)
    results = []
    while (done < per_game).any():
        obs = env.observe()
        status = env.step(policy(obs, rng))
        for g in np.flatnonzero(status):
            if done[g] < per_game:
                results.append(json.loads(env.last_episode(int(g))))
            done[g] += 1
    results = results[:fights]
    return summarize(results)


def summarize(results):
    won = np.array([r["won"] for r in results])
    hp_lost = np.array([80 - r["hp"] for r in results])
    turns = np.array([r["turns"] for r in results])
    return {
        "fights": len(results),
        "win_rate": float(won.mean()),
        "hp_lost": float(hp_lost.mean()),
        "hp_lost_when_won": float(hp_lost[won].mean()) if won.any() else None,
        "turns": float(turns.mean()),
        "timed_out": int(sum(r.get("timed_out", False) for r in results)),
        "win_rate_by_act": {
            str(act): float(np.mean([r["won"] for r in results if ACT_OF.get(r.get("encounter")) == act]))
            for act in (1, 2, 3, 4)
            if any(ACT_OF.get(r.get("encounter")) == act for r in results)
        },
    }


def main():
    fights = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    sets = {
        "random_decks_all_encounters": dict(seed=100_000),
        "starter_deck_act1": dict(seed=200_000, random_decks=False, encounters=ACT1),
    }
    report = {}
    for set_name, kwargs in sets.items():
        for name, policy in (("random", random_policy), ("scripted", scripted_policy)):
            t = time.time()
            r = evaluate(policy, fights, **kwargs)
            r["seconds"] = round(time.time() - t, 1)
            report[f"{set_name}/{name}"] = r
            acts = "  ".join(f"A{k} {v:.0%}" for k, v in r["win_rate_by_act"].items())
            print(f"{set_name:28} {name:9} win {r['win_rate']:6.1%}  hp lost {r['hp_lost']:5.1f}  turns {r['turns']:4.1f}  [{acts}]  ({r['seconds']}s)")
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "baselines.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
