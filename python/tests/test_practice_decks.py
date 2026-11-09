"""Checks for --practice-decks (plain asserts, no pytest needed):

    python python/tests/test_practice_decks.py

The Defect's practice fights from crates/sts-py practice.rs build_deck come in every deck kind and
act, train_runs.practice_summary breaks their wins down by kind, act and room and counts the HP the
won ones cost, evaluate_practice plays a fixed set of each kind in each act (and with bosses=True
the same at each act's boss), and evaluate_scenarios plays human boss fights of every act, the
Heart included (here with a tiny untrained network on the CPU)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import contextlib

import numpy as np
import torch

from core.run_model import RunNet
from core.sts_env import RunEnv
from training.train_runs import DECK_ACTS, DECK_KINDS, deck_kind, evaluate_practice, evaluate_scenarios, hp_lost, practice_summary, retry_summary


def main():
    env, rng = RunEnv(16, seed=4, practice=1.0, practice_decks="mix", character="defect"), np.random.default_rng(4)
    eps = []
    while len(eps) < 120:
        obs = env.observe()
        status = env.step([rng.choice(np.flatnonzero(m)) for m in obs["mask"]])
        eps += [json.loads(env.last_episode(int(g))) for g in np.flatnonzero(status)]
    kinds = {deck_kind(e["practice"]) for e in eps}
    acts = {e["act"] for e in eps}
    assert kinds == set(DECK_KINDS), kinds
    assert acts == set(DECK_ACTS), acts
    s = practice_summary(eps)
    for k in DECK_KINDS + [f"act{a}" for a in DECK_ACTS]:
        assert s[f"practice_win_{k}"] != "", (k, s)
    assert s["practice_win_block"] == ""
    assert all(1 <= e["hp_start"] <= e["max_hp"] for e in eps), "every fight's starting HP"
    assert {e["room"] for e in eps} == {"Monster", "Elite", "Boss"}, {e["room"] for e in eps}
    assert all(s[f"practice_win_room_{r}"] != "" for r in ("hallway", "elite", "boss")), s
    won = [e for e in eps if e["won"]]
    assert won and abs(s["practice_hp_lost"] - np.mean([e["hp_start"] - e["hp"] for e in won])) < 1e-9, s
    fake = [{"won": True, "hp_start": 50, "hp": 40, "max_hp": 80, "act": 1}, {"won": True, "hp_start": 60, "hp": 30, "max_hp": 60, "act": 2},
            {"won": False, "hp_start": 70, "hp": 0, "max_hp": 70, "act": 2}]
    h = hp_lost(fake, "x_")
    assert (h["x_hp_lost"], h["x_hp_lost_share"], h["x_hp_lost_act1"], h["x_hp_lost_act2"], h["x_hp_lost_act3"]) == (20.0, 0.3125, 10.0, 30.0, ""), h

    torch.manual_seed(0)
    net = RunNet(d=16, layers=1, heads=2)
    row = evaluate_practice(net, torch.device("cpu"), 2, contextlib.nullcontext, ascension=10, character="defect", decks="mix")
    assert row["runs"] == 2 * len(DECK_KINDS) * len(DECK_ACTS), row
    for k in DECK_KINDS + [f"act{a}" for a in DECK_ACTS]:
        assert 0.0 <= row[f"win_{k}"] <= 1.0, (k, row)
    assert row["hp_lost"] == "" or row["hp_lost"] >= -30, row  # end-of-fight healing can make it negative
    again = evaluate_practice(net, torch.device("cpu"), 2, contextlib.nullcontext, ascension=10, character="defect", decks="mix")
    assert again == row, "the eval's fights are fixed"
    boss = evaluate_practice(net, torch.device("cpu"), 2, contextlib.nullcontext, character="defect", decks="mix", bosses=True)
    assert boss["set"] == "practice_boss" and boss["runs"] == row["runs"], boss
    assert boss["win_room_boss"] != "" and boss.get("win_room_hallway", "") == "" and boss.get("win_room_elite", "") == "", boss

    deck = [["Strike_B", 0]] * 4 + [["Defend_B", 1]] * 4 + [["Zap", 1], ["Dualcast", 0], ["Glacier", 1], ["Defragment", 1], ["Coolheaded", 0]]
    fights = [(1, 16, ["Hexaghost"]), (2, 33, ["Champ"]), (3, 50, ["Time Eater", "Donu and Deca"]), (4, 55, ["The Heart"])]
    lines = [json.dumps({"character": "DEFECT", "ascension": 20, "act": act, "floor": floor, "hp": 60, "max_hp": 70, "bosses": bosses,
                         "deck": deck, "relics": ["Cracked Core", "Vajra"], "potions": ["Regen Potion"]}) for act, floor, bosses in fights]
    sc = evaluate_scenarios(net, torch.device("cpu"), lines, contextlib.nullcontext, batch=4, character="defect")
    assert sc["runs"] == 4 and all(sc[f"win_boss{a}"] != "" for a in (1, 2, 3, 4)), sc
    print(f"practice decks: ok ({len(eps)} training fights; eval {row['runs']} fights, won {row['win_rate']:.0%} untrained)")
    retries(lines)


def retries(lines):
    """--retry-losses 3 on practice decks and human boss scenarios, with random moves."""
    env, rng = RunEnv(16, seed=6, practice=1.0, practice_decks="mix", character="defect"), np.random.default_rng(6)
    env.envs.set_scenarios(lines, 0.5, False)
    env.envs.set_retry_losses(3)
    eps, last = [], {}
    while len(eps) < 300:
        obs = env.observe()
        seeds = [env.envs.episode_seed(g) for g in range(16)]
        status = env.step([rng.choice(np.flatnonzero(m)) for m in obs["mask"]])
        for g in np.flatnonzero(status):
            e = json.loads(env.last_episode(int(g)))
            e["seed"] = seeds[g]
            prev = last.get(int(g))
            if prev is not None:
                again = not prev["won"] and prev["retry"] < 3
                assert e["retry"] == (prev["retry"] + 1 if again else 0), (prev, e)
                if again:
                    assert (e["seed"], e["encounter"], e.get("scenario"), e["hp_start"]) == (prev["seed"], prev["encounter"], prev.get("scenario"), prev["hp_start"]), (prev, e)
            last[int(g)] = e
            eps.append(e)
    replays = [e for e in eps if e["retry"]]
    assert replays and any(e.get("scenario") is not None for e in replays), "replays of decks and of human fights"
    s = retry_summary(eps[-16:], eps)
    assert s["retry_fights"] == sum(1 for e in eps[-16:] if e["retry"]) and 0 <= s["retry_win"] <= 1, s
    assert s["retry_saved"] == sum(e["won"] for e in replays) / sum(1 for e in eps if not e["retry"] and not e["won"]), s
    print(f"retries: ok ({len(replays)} replays of {len(eps)} fights, {s['retry_win']:.0%} won)")


if __name__ == "__main__":
    main()
