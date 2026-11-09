"""Checks for playing on from human boss states (plain asserts, no pytest needed):

    python python/tests/test_play_on.py

human_bosses.scenarios on made-up runs: without flags only winners, unlabelled, as ever; --clean leaves the cheats out and puts unexplained cards in
where the boss relic gave them; --all-outcomes keeps the rest too, with what became of each. Envs.set_scenarios(play_on=True): not a practice fight,
and the summary is a run's naming its scenario. analysis/play_on.py: the rates; its own arrivals from a dump; a --fighter on a plain run network
really fights (search_eval.py --fighter silently doesn't); a few runs from small random networks, in the lines' order."""
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import numpy as np
import sts
import torch

from analysis.play_on import adjust, choose, make_policy, measures, own_arrivals, own_runs, play_states, read_lines, reading
from core.fight_model import masked
from core.run_model import RunNet
from core.split import SplitPolicy
from core.sts_env import RunEnv
from human import human_bosses as hb

STARTER = ["Strike_G"] * 5 + ["Defend_G"] * 5 + ["Survivor", "Neutralize", "AscendersBane"]


def human_run(floor_reached, victory=False, killed_by=None, **more):
    """A made-up A20 Silent run that fought both Act 1's and Act 2's bosses if it got that far."""
    enemies = {16: "Hexaghost", 33: "Champ", 50: "Time Eater", 51: "Donu and Deca"}
    return {"play_id": f"id-{floor_reached}-{victory}-{killed_by}", "character_chosen": "THE_SILENT", "build_version": "2020-07-30", "ascension_level": 20,
            "floor_reached": floor_reached, "victory": victory, "killed_by": killed_by, "master_deck": STARTER + ["Backstab"], "relics": ["Ring of the Snake", "Vajra"],
            "card_choices": [{"floor": 3, "picked": "Backstab", "not_picked": ["Slice", "Dash"]}], "relics_obtained": [{"floor": 9, "key": "Vajra"}],
            "damage_taken": [{"floor": f, "enemies": e} for f, e in enemies.items() if f <= floor_reached],
            "current_hp_per_floor": [40] * floor_reached, "max_hp_per_floor": [70] * floor_reached, "gold_per_floor": [120] * floor_reached,
            "path_per_floor": ["M"] * floor_reached, "items_purged": ["Strike_G"], "items_purged_floors": [20], **more}


def line_of(sc):
    return json.dumps(sc)


# Without flags: winners only, no label, as ever. A cheat among them stays.
winner, heart, died2, died1 = human_run(52, True), human_run(57, True), human_run(25, killed_by="Slavers"), human_run(16, killed_by="Hexaghost")
abandoned, cheat = human_run(20), human_run(52, True, gold_per_floor=[1500] + [120] * 51)
cheat["play_id"] = "cheat"
kinds = ("boss1", "boss2")
assert hb.scenarios(died2, 20, "THE_SILENT", kinds) == [] and hb.scenarios(abandoned, 20, "THE_SILENT", kinds) == []
plain = hb.scenarios(winner, 20, "THE_SILENT", kinds)
assert [s["act"] for s in plain] == [1, 2] and all("outcome" not in s and "gold" not in s for s in plain)
assert len(hb.scenarios(cheat, 20, "THE_SILENT", kinds)) == 2
# --clean: the cheat and abandons go; the winner's line is the same.
assert hb.scenarios(cheat, 20, "THE_SILENT", kinds, clean=True) == []
assert hb.scenarios(winner, 20, "THE_SILENT", kinds, clean=True) == plain
# --all-outcomes: everyone who fought that boss, labelled, with gold and removals so far; a death at a boss has that boss's state.
everyone = {r["play_id"]: hb.scenarios(r, 20, "THE_SILENT", kinds, clean=True, all_outcomes=True) for r in (winner, heart, died2, died1, abandoned, cheat)}
assert everyone[abandoned["play_id"]] == [] and everyone[cheat["play_id"]] == []
assert [s["outcome"]["label"] for s in everyone[winner["play_id"]]] == ["won", "won"]
assert [s["outcome"]["label"] for s in everyone[heart["play_id"]]] == ["heart", "heart"]
assert everyone[died2["play_id"]][0]["outcome"] == {"label": "died", "floor_reached": 25, "died_act": 2, "killed_by": "Slavers"}
assert len(everyone[died2["play_id"]]) == 1  # no Act 2 boss fought
assert [s["bosses"] for s in everyone[died1["play_id"]]] == [["Hexaghost"]] and everyone[died1["play_id"]][0]["outcome"]["died_act"] == 1
assert [(s["gold"], s["purges"], s["hp"]) for s in everyone[winner["play_id"]]] == [(120, 0, 40), (120, 1, 40)], everyone[winner["play_id"]]
# --clean's fill: unexplained cards wait for the boss relic that gave them (Pandora's Box here, Astrolabe, Empty Cage); without it they're in at floor 0.
late = human_run(52, True, master_deck=["Survivor", "Neutralize", "AscendersBane", "Backstab", "Malaise", "Die Die Die"], boss_relics=[{"picked": "Pandora's Box", "not_picked": []}])
early = {s["act"]: sorted(c for c, _ in s["deck"]) for s in hb.scenarios(late, 20, "THE_SILENT", kinds)}
fixed = {s["act"]: sorted(c for c, _ in s["deck"]) for s in hb.scenarios(late, 20, "THE_SILENT", kinds, clean=True)}
assert "Malaise" in early[1], early[1]
assert "Malaise" not in fixed[1] and "Strike_G" in fixed[1], fixed[1]
assert fixed[2] == sorted(late["master_deck"]), fixed[2]
# --clean's relics: the one Neow gave is in the state from the start, a boss relic in the starter's place. Without it, as ever.
swapped = human_run(52, True, neow_bonus="BOSS_RELIC", relics=["Cursed Key", "Vajra", "Potion Belt"], relics_obtained=[{"floor": 9, "key": "Vajra"}, {"floor": 20, "key": "Potion Belt"}])
assert [s["relics"] for s in hb.scenarios(swapped, 20, "THE_SILENT", kinds)] == [["Ring of the Snake", "Vajra"], ["Ring of the Snake", "Vajra", "Potion Belt"]]
assert [s["relics"] for s in hb.scenarios(swapped, 20, "THE_SILENT", kinds, clean=True)] == [["Cursed Key", "Vajra"], ["Cursed Key", "Vajra", "Potion Belt"]]
rare = human_run(52, True, neow_bonus="ONE_RARE_RELIC", relics=["Ring of the Snake", "Girya", "Vajra"])
assert [s["relics"] for s in hb.scenarios(rare, 20, "THE_SILENT", kinds, clean=True)] == [["Ring of the Snake", "Girya", "Vajra"]] * 2
assert sts.scenario_problem(json.dumps(hb.scenarios(swapped, 20, "THE_SILENT", kinds, clean=True)[0])) is None

# Play on: the same line, a practice fight or a normal run.
line = line_of(everyone[died1["play_id"]][0])
assert sts.scenario_problem(line) is None
assert "unknown fight" in sts.scenario_problem(line.replace("Hexaghost", "Champ"))
for play_on in (False, True):
    env = RunEnv(2, seed=9, ascension=20, practice=1.0, max_steps=8000, character="silent")
    assert env.envs.set_scenarios([line], 1.0, True, play_on) == 0
    assert list(env.envs.practice_mask()) == [not play_on] * 2
    ended = False
    for step in range(20000):
        obs = env.observe()
        status = env.step(np.array([int(np.random.default_rng(step + g).choice(np.flatnonzero(obs["mask"][g]))) for g in range(2)]))
        if status[0]:
            e = json.loads(env.last_episode(0))
            ended = True
            break
    assert ended
    assert ("practice" in e) == (not play_on) and ("keys" in e) == play_on and e["scenario"] == 0, e
    assert e["floor"] >= 16

# The rates: from summaries, and from a dump's own arrivals.
eps = [{"won": False, "act": 1, "floor": 16}, {"won": False, "act": 2, "floor": 25, "keys": 1}, {"won": False, "act": 3, "floor": 40, "keys": 2},
       {"won": True, "act": 3, "floor": 52, "keys": 1}, {"won": False, "act": 4, "floor": 55, "keys": 3}, {"won": True, "act": 4, "floor": 57, "keys": 3}]
m = measures(eps, 16, 1)
assert (m["n"], m["beat"][0], m["next"][0], m["act3"][0], m["pair"][0]) == (6, 5 / 6, 5 / 6, 4 / 6, 3 / 6)
assert m["heart"][0] == 1 / 6 and m["three_keys"] == 2 / 6 and abs(m["pair"][1] - (0.25 / 6) ** 0.5) < 1e-9
m2 = measures(eps, 33, 2)
assert (m2["beat"][0], m2["next"][0]) == (4 / 6, 4 / 6)
tmp = tempfile.mkdtemp()
for i, part in enumerate((eps[:3], eps[3:] + [{"won": False, "act": 1, "floor": 6, "died_in": "Gremlin Nob", "keys": 0}])):
    with open(f"{tmp}/own.{i}.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in part)
runs = own_runs(f"{tmp}/own")
assert len(runs) == 7 and len(own_arrivals(runs, 16)) == 6 and len(own_arrivals(runs, 33)) == 4
assert len(own_runs(f"{tmp}/own,{tmp}/own")) == 14
winners = measures(eps, 16, 1)
said = reading({"winners": winners, "act2-deaths": measures(eps[:2], 16, 1)}, measures(eps[:4], 16, 1), ["winners", "act2-deaths"])
assert len(said) == 2 and said[0].strip().startswith("next act: winners 83.3; 1.11 ±0.38 x its own 75.0; +33.3 on act 2 deaths"), said
said = reading({"winners": measures(eps[:4], 16, 1), "act2-deaths": measures(eps[:2], 16, 1)}, None, ["winners", "act2-deaths"], "winners never practised")
assert said[0].strip().startswith("next act: winners never practised 75.0; +25.0 on act 2 deaths"), said

# Choosing lines: labels into groups, held out of training first; a stripped state keeps her deck.
with open(f"{tmp}/lines.jsonl", "w") as f:
    for i in range(200):
        s = dict(everyone[winner["play_id"]][0], play_id=f"p{i}", outcome={"label": "won" if i % 3 else "heart", "floor_reached": 52})
        f.write(json.dumps(s) + "\n")
    s = dict(everyone[winner["play_id"]][1], play_id="p-act2", outcome={"label": "died", "floor_reached": 33, "died_act": 2, "killed_by": "Champ"})
    f.write(json.dumps(s) + "\n")
cells, counts = read_lines([f"{tmp}/lines.jsonl"], {"boss1", "boss2"}, 5)
chosen, found = choose(cells, counts, "boss1", "winners", 5)
assert found == 200 and len(chosen) == 5
ids = [json.loads(x)["play_id"] for x in chosen]
held = {f"p{j}" for j in range(200) if hb.held_out(f"p{j}")}
assert len(held) >= 2 and all(i in held for i in ids[:len(held)]) and not any(i in held for i in ids[len(held):]), (ids, held)
assert choose(cells, counts, "boss2", "act2-deaths", 5)[1] == 1 and choose(cells, counts, "boss2", "winners", 5) == ([], 0)
s = json.loads(adjust(chosen[0], True, False))
assert s["relics"] == ["Ring of the Snake"] and s["deck"] == json.loads(chosen[0])["deck"] and s["hp"] == 40
assert json.loads(adjust(chosen[0], False, True))["hp"] == 70 and json.loads(adjust(chosen[0], False, False)) == json.loads(chosen[0])

# --fighter on a plain run network: the fights are the fighter's, the choices the network's.
device = torch.device("cpu")


def amp():
    return contextlib.nullcontext()


def small(path, seed, **args):
    torch.manual_seed(seed)
    torch.save({"net": RunNet(d=16, layers=1, heads=2).state_dict(), "args": {"d": 16, "layers": 1, "heads": 2, **args}, "update": 7}, path)


small(f"{tmp}/plain.pt", 1)
small(f"{tmp}/fighter.pt", 2)
small(f"{tmp}/split.pt", 3, fight_net=f"{tmp}/fighter.pt")


class Args:
    checkpoint, fighter, batch, seed, character, free_keys = f"{tmp}/plain.pt", None, 2, 700_000, "silent", False


args = Args()
alone, lookup, who = make_policy(args, device, amp)
assert isinstance(alone, RunNet) and lookup is None and "plays everything" in who
args.fighter = f"{tmp}/fighter.pt"
split, lookup, who = make_policy(args, device, amp)
assert isinstance(split, SplitPolicy) and lookup is None and who.startswith(f"fights: {tmp}/fighter.pt")
args.checkpoint, args.fighter = f"{tmp}/split.pt", None
own_split, _, who = make_policy(args, device, amp)
assert isinstance(own_split, SplitPolicy) and who.startswith(f"fights: {tmp}/fighter.pt")
env = RunEnv(1, seed=9, ascension=20, practice=1.0, character="silent")
env.envs.set_scenarios([line], 1.0, True, True)
obs = env.observe()
ints, floats = torch.from_numpy(obs["ints"]).long(), torch.from_numpy(obs["floats"])
with torch.no_grad():
    a, _ = split(ints, floats, torch.zeros(1, dtype=torch.bool))
    b, _ = alone(ints, floats, torch.zeros(1, dtype=torch.bool))
    c, _ = split.fight(ints, floats, torch.zeros(1, dtype=torch.bool))
assert torch.equal(masked(a.float(), torch.from_numpy(obs["mask"])), masked(c.float(), torch.from_numpy(obs["mask"])))
assert not torch.equal(a, b)

# A few runs from small random networks: one summary a state, in the lines' order; on HP this high they go through a boss at least sometimes.
lines = [line_of(dict(everyone[winner["play_id"]][0], hp=900, max_hp=900, play_id=f"p{i}")) for i in range(5)]
with contextlib.redirect_stdout(io.StringIO()):
    out = play_states(split, None, device, amp, lines, args, "test")
assert [e["scenario"] for e in out] == list(range(5)) and all("practice" not in e for e in out)
assert all(e["floor"] >= 16 for e in out), out

print("ok")
