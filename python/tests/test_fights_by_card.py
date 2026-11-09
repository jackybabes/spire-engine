"""Checks for analysis/seed_rates.py and analysis/fights_by_card.py (plain asserts, no pytest needed):

    python python/tests/test_fights_by_card.py

The control seed: a forced seed's runs all start with its cards from the first episode (set_seeds alone holds from the next
one), and the control is drawn only when named. seed_rates' policy: a plain network plays alone, --fighter or args.fight_net
fights instead, a --deck-net checkpoint is refused. A scenario line's "seed" gives the same draws in any game. fights_by_card:
one copy removed, the least upgraded; fights sampled evenly, each with its seed, those it can't set up left out; `play_fights`
plays each fight once and in order; `Plays` on scripted fights of Deadly Poison and Catalyst (on the first monster and on
another, the poison read from the engine's JSON view) and one of Acrobatics and Tactician counts what was done."""
import contextlib
import gzip
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import numpy as np
import torch

import sts
from analysis.fights_by_card import LAYOUT, MAX_RELICS, PICK, POTION, TARGETS, Plays, card_ids, evenly, kind_of, play_fights, read_fights, report_archetypes, without
from analysis.seed_rates import load_policy, seeded_runs
from core.run_model import RunNet
from core.split import SplitPolicy
from core.sts_env import RunEnv
from training.train_runs import in_combat


def amp():
    return contextlib.nullcontext()


FIGHT = {"character": "THE_SILENT", "ascension": 20, "act": 1, "floor": 16, "hp": 60, "max_hp": 70, "bosses": ["Hexaghost"],
         "deck": [["Strike_G", 0], ["Defend_G", 0], ["Backflip", 1], ["Backflip", 0], ["Backflip", 1]], "relics": ["Ring of the Snake"], "potions": []}


def tiny_checkpoint(folder, name, **args):
    path = os.path.join(folder, f"{name}.pt")
    a = {"character": "silent", "d": 16, "layers": 1, "heads": 2, **args}
    net = RunNet(d=16, layers=1, heads=2)
    torch.save({"net": net.state_dict(), "args": a, "update": 1}, path)
    return path


def test_seeds():
    names = {"control": [("Dash", True), ("Predator", False)], "poison": [("Noxious Fumes", True), ("Catalyst", False)]}
    for name, cards in names.items():
        env = RunEnv(8, seed=5, ascension=20, max_steps=50, character="silent", seeded=1.0)
        env.envs.set_seeds([name])
        env.envs.reset()
        for g in range(8):
            deck = json.loads(env.envs.observe(g))["deck"]
            for card, upgraded in cards:
                assert any(c["name"].rstrip("+") == card and c["upgraded"] == upgraded for c in deck), (name, card, deck)
    # Not named, it never comes (64 first episodes).
    env = RunEnv(64, seed=5, ascension=20, max_steps=50, character="silent", seeded=1.0)
    decks = [json.loads(env.envs.observe(g))["deck"] for g in range(64)]
    assert not any(c["id"] == "Predator" for d in decks for c in d) and any(c["id"] == "Catalyst" for d in decks for c in d)
    # Free cards instead of a named seed, and the three keys from the start.
    env = RunEnv(4, seed=5, ascension=20, max_steps=50, character="silent", seeded=1.0)
    env.envs.set_seed_cards(["Catalyst+", "Blade Dance"])
    env.envs.set_free_keys(True)
    env.envs.reset()
    for g in range(4):
        view = json.loads(env.envs.observe(g))
        assert any(c["name"].rstrip("+") == "Catalyst" and c["upgraded"] for c in view["deck"]) and any(c["name"] == "Blade Dance" and not c["upgraded"] for c in view["deck"]), view["deck"]
    try:
        env.envs.set_seed_cards(["Catalsyt"])
        raise AssertionError("a misspelt card was accepted")
    except ValueError:
        pass
    # A seed can wait for an act: not in the deck at the start.
    env = RunEnv(4, seed=5, ascension=20, max_steps=50, character="silent", seeded=1.0)
    env.envs.set_seed_cards(["Catalyst"])
    env.envs.set_seed_act(2)
    env.envs.reset()
    assert not any(c["id"] == "Catalyst" for g in range(4) for c in json.loads(env.envs.observe(g))["deck"])
    print("seeds: forced from the first episode, the control only when named, free cards by name, waiting for an act")


def test_seeded_runs():
    with tempfile.TemporaryDirectory() as folder:
        plain, fighter = tiny_checkpoint(folder, "plain"), tiny_checkpoint(folder, "fighter")
        deck_net = tiny_checkpoint(folder, "deck", fight_net=fighter)
        device = torch.device("cpu")
        net, lookup, _ = load_policy(plain, device, amp)
        assert isinstance(net, RunNet) and lookup is None
        for policy, _, _ in (load_policy(plain, device, amp, fighter), load_policy(deck_net, device, amp)):
            assert isinstance(policy, SplitPolicy)
        # A fight network isn't what makes the choices: refused if its deck network did, a warning if nothing did.
        try:
            load_policy(tiny_checkpoint(folder, "trained-with-deck-net", deck_net=plain), device, amp)
            raise AssertionError("a --deck-net checkpoint plays on its own stale choices")
        except SystemExit as e:
            assert "--fighter" in str(e)
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            load_policy(tiny_checkpoint(folder, "fights-only", practice=1.0), device, amp)
        assert "WARNING" in said.getvalue()
        with contextlib.redirect_stdout(io.StringIO()):
            out = seeded_runs(net, None, device, amp, "control", 3, 2, 1000, 20, "silent")
            bare = seeded_runs(net, None, device, amp, None, 3, 2, 1000, 20, "silent")
    assert [e["seed"] for e in out] == [1000, 1001, 1002] == [e["seed"] for e in bare]
    assert all(e["seeded"] == "control" for e in out) and all(e["seeded"] is None for e in bare)
    print("seed_rates: the same seeds in every line, the seed forced")


def draw_order(line, game):
    env = sts.Envs(2, mode="run", seed=game, practice=1.0, character="silent")
    assert env.set_scenarios([line, line], 1.0, True) == 0
    views = [json.loads(env.observe(g))["screen"]["combat"] for g in range(2)]
    return [([c["name"] for c in v["hand"]], [m["hp"] for m in v["monsters"]]) for v in views]


def test_line_seed():
    line = json.dumps(dict(FIGHT, seed=4242, deck=[[c, 0] for c in ("Strike_G", "Defend_G", "Backflip", "Dash", "Catalyst", "Footwork", "Acrobatics", "Prepared", "Slice", "Burst")]))
    a, b = draw_order(line, 1), draw_order(line, 90)
    assert a[0] == a[1] == b[0], "the line's seed, whichever game plays it"
    plain = json.dumps({k: v for k, v in json.loads(line).items() if k != "seed"})
    c = draw_order(plain, 1)
    assert c[0] != c[1], "otherwise the game's"
    print("scenario seed: the same draws in any game")


def test_fights():
    assert [c[:2] for c in without(FIGHT, "Backflip")["deck"]] == [["Strike_G", 0], ["Defend_G", 0], ["Backflip", 1], ["Backflip", 1]], "the least upgraded"
    assert len(FIGHT["deck"]) == 5, "the fight itself is left alone"
    assert evenly(list(range(10)), 5) == [0, 2, 4, 6, 8] and evenly(list(range(3)), 5) == [0, 1, 2] and evenly(list(range(3)), 0) == [0, 1, 2]
    assert kind_of(FIGHT) == "boss1" and kind_of(dict(FIGHT, room="elite", act=2)) == "elite2"
    assert LAYOUT["card_java_ids"][card_ids(["Well-Laid Plans"])[0]] == "Well Laid Plans" and LAYOUT["card_java_ids"][card_ids(["Strike_G"])[0]] == "Strike_G"
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "lines.jsonl.gz")
        with gzip.open(path, "wt") as f:
            f.write(json.dumps(FIGHT) + "\n" + json.dumps(dict(FIGHT, relics=["Ring of the Snake"] * MAX_RELICS)) + "\n")
            f.write(json.dumps(dict(FIGHT, deck=[["No Such Card", 0]])) + "\n" + json.dumps(dict(FIGHT, room="elite", bosses=["Lagavulin"])) + "\n")
        with contextlib.redirect_stdout(io.StringIO()):
            fights = read_fights(path, "silent", None, 0)
            elites = read_fights(path, "silent", {"elite1"}, 0)
    assert [f["seed"] - fights[0]["seed"] for f in fights] == [0, 3], "cheated and unknown left out, the seeds by place in the file"
    assert [kind_of(f) for f in elites] == ["elite1"]

    net = RunNet(d=16, layers=1, heads=2).eval()
    decks = [dict(FIGHT, seed=7000 + i) for i in range(5)]
    eps = play_fights(net, torch.device("cpu"), amp, decks, 2, "silent")
    assert [e["scenario"] for e in eps] == list(range(5)) and all(e["practice"] == "boss1" for e in eps)
    print("fights_by_card: removal, sampling, the fights played once and in order")


def scripted_plays(deck, cards, script, max_steps=60, **fight):
    """Plays a fight of `deck` for `max_steps` moves by `script(obs, env) -> slot`; `Plays` of it, ended."""
    env = RunEnv(1, seed=1, practice=1.0, max_steps=4000, character="silent")
    line = {**FIGHT, "hp": 70, "deck": deck, "seed": 11, "bosses": ["Hexaghost"], **fight}
    assert env.envs.set_scenarios([json.dumps(line)], 1.0, True) == 0
    plays = Plays(card_ids(cards), 1)
    for _ in range(max_steps):
        obs = env.observe()
        if not in_combat(obs["ints"])[0]:
            break
        acts = np.array([script(obs, env)])
        plays.update(obs, acts, np.array([True]))
        status = env.step(acts)
        plays.end(status > 0)
        if status[0]:
            return plays
    plays.end(np.array([True]))
    return plays


def poison_fight(target, **fight):
    """A scripted fight of Deadly Poison, then Catalyst, on monster slot `target` (Footwork in hand, never played): its `Plays`,
    checked against the engine's own view (not the arrays `Plays` reads): the poison on that monster at each play, and
    which cards the legal moves can play each turn."""
    deadly, catalyst = [i + 1 for i in card_ids(["Deadly Poison", "Catalyst"])]
    log = []
    playable = {}

    def poison_first(obs, env):
        view = json.loads(env.envs.observe(0))["screen"]["combat"]
        for move in map(json.loads, env.envs.legal_actions(0)):
            if move["action"]["type"] == "play_card":
                playable.setdefault(view["turn"], set()).add(view["hand"][move["action"]["hand_index"]]["id"])
        mask = obs["mask"][0]
        hand = obs["i_hand_card"][0]
        for card in (deadly, catalyst):
            for h in np.flatnonzero(hand == card):
                slot = h * TARGETS + target
                if mask[slot]:
                    monsters = [m for m in view["monsters"] if m["alive"]]
                    log.append((card, sum(p["amount"] for p in monsters[target - 1]["powers"] if p["id"] == "Poison")))
                    return slot
        return LAYOUT["actions"]["end_turn"]

    names = ["Deadly Poison", "Catalyst", "Footwork", "Dash"]
    plays = scripted_plays([["Deadly Poison", 0]] * 5 + [["Catalyst", 0]] * 3 + [["Footwork", 0]], names, poison_first, **fight)
    r = plays.report(names)
    played = [sum(c == x for c, _ in log) for x in (deadly, catalyst)]
    assert played[0] and played[1], log
    assert r["cards"]["Deadly Poison"]["plays"] == played[0] and r["cards"]["Catalyst"]["plays"] == played[1]
    for name in names[:3]:
        assert r["cards"][name]["turns"] == sum(name in cards for cards in playable.values()), (name, r["cards"][name], playable)
    assert r["cards"]["Footwork"]["turns"] > 0 and r["cards"]["Footwork"]["turns_played"] == 0 and r["cards"]["Footwork"]["fights_played"] == 0
    assert r["cards"]["Dash"] == {"turns": 0, "turns_played": 0, "fights": 0, "fights_played": 0, "plays": 0}
    assert r["cards"]["Catalyst"]["fights"] == 1 and r["cards"]["Catalyst"]["fights_played"] == 1
    assert 0 < r["cards"]["Catalyst"]["turns_played"] <= r["cards"]["Catalyst"]["turns"]
    cat = r["catalyst"]
    assert sum(cat["plays_on"]) == played[1]
    on = [int(np.digitize(p, [1, 5, 10, 20])) for c, p in log if c == catalyst]
    assert max(on) > 0, ("Catalyst only ever on no poison: the test shows nothing", log)
    assert cat["plays_on"] == [on.count(i) for i in range(5)], (cat, log)
    assert sum(cat["turns"]) == r["cards"]["Catalyst"]["turns"] and sum(cat["turns_played"]) == r["cards"]["Catalyst"]["turns_played"]
    return cat, played[1]


def test_report_archetypes():
    poison = dict(FIGHT, deck=[["Catalyst", 0], ["Noxious Fumes", 0], ["Deadly Poison", 0]])
    wraith = dict(FIGHT, deck=[["Wraith Form v2", 0], ["Strike_G", 0]], bosses=["The Heart"], act=4, floor=56)
    with contextlib.redirect_stdout(io.StringIO()):
        table = report_archetypes([poison, poison, wraith], [{"won": True}, {"won": False}, {"won": True}])
    assert table["poison"][kind_of(poison)] == [2, 1.0], table
    assert table["wraith form"][kind_of(wraith)] == [1, 1.0], table
    assert set(table) == {"poison", "wraith form"}, table


def test_heart_add():
    from analysis.heart_add import by_deck, fights_of, java_card, own_decks, usable
    assert java_card("Strike") == ["Strike_G", 0] and java_card("Wraith Form+") == ["Wraith Form v2", 1] and java_card("Cloak and Dagger") == ["Cloak And Dagger", 0]
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "x.0.jsonl"), "w") as f:
            f.write(json.dumps({"floor": 56, "act": 4, "won": False, "max_hp": 70, "deck": ["Strike", "Defend+", "Wraith Form+"], "relics": ["Ring of the Snake"], "log": {}}) + "\n")
            f.write(json.dumps({"floor": 50, "act": 3, "won": False, "max_hp": 70, "deck": ["Strike"], "relics": []}) + "\n")
        decks = own_decks([os.path.join(d, "x")])
    assert len(decks) == 1 and decks[0]["deck"] == [["Strike_G", 0], ["Defend_G", 1], ["Wraith Form v2", 1]]
    env = sts.Envs(1, mode="run", practice=1.0, character="silent")
    assert usable(env, decks) == decks
    assert usable(env, [dict(decks[0], relics=["No Such Relic"])]) == []
    fights = fights_of(decks * 2, 2, 0.5, [["Catalyst", 1]])
    assert len(fights) == 4 and fights[0]["hp"] == 35 and fights[0]["deck"][-1] == ["Catalyst", 1] and decks[0]["deck"][-1] != ["Catalyst", 1]
    assert fights_of(decks * 2, 2, 0.5)[3]["seed"] == fights[3]["seed"] and len({f["seed"] for f in fights}) == 4, "the same seeds, whatever is added"
    assert by_deck([{"won": True}, {"won": False}, {"won": True}, {"won": True}], 2).tolist() == [0.5, 1.0]
    print("heart_add: decks from dumps, the fights pair up")


def test_plays():
    cat, played = poison_fight(1)
    assert cat["on_most_poison"] == played
    # Another monster than the first: the target's slot is read as the target's poison.
    poison_fight(2, room="elite", act=2, floor=25, bosses=["Slavers"])

    acrobatics, tactician = [i + 1 for i in card_ids(["Acrobatics", "Tactician"])]
    choices = []

    def discard_tactician(obs, env):
        # Acrobatics, then Tactician at the selection if it's on offer (else the first option), and ends the turn.
        mask = obs["mask"][0]
        if obs["i_choice_kind"][0, 0]:
            offers = obs["i_option_card"][0]
            on = [k for k in range(64) if mask[PICK + k] and offers[k] == tactician]
            choices.append(bool(on))
            return PICK + (on[0] if on else int(np.flatnonzero(mask[PICK:PICK + 64])[0]))
        for h in np.flatnonzero(obs["i_hand_card"][0] == acrobatics):
            if mask[h * TARGETS]:
                return h * TARGETS
        return LAYOUT["actions"]["end_turn"]

    plays = scripted_plays([["Acrobatics", 0]] * 5 + [["Tactician", 0]] * 3 + [["Strike_G", 0]] * 2, ["Acrobatics", "Tactician"], discard_tactician)
    r = plays.report(["Acrobatics", "Tactician"])
    assert choices and any(choices)
    assert r["cards"]["Tactician"]["turns"] == 0, "Tactician can't be played"
    assert r["cards"]["Tactician"]["selections"]["Discard"] == {"offered": sum(choices), "picked": sum(choices)}
    s = r["selections"]["Discard"]
    assert s["decisions"] == len(choices) and s["stops"] == 0 and s["picked"]["Tactician or Reflex"] == sum(choices)
    assert abs(sum(s["at_random"].values()) - len(choices)) < 1e-9
    json.dumps(r)
    print("Plays: the scripted plays, Catalyst's poison and the discards counted")


if __name__ == "__main__":
    test_seeds()
    test_seeded_runs()
    test_line_seed()
    test_fights()
    test_report_archetypes()
    test_heart_add()
    test_plays()
    print("ok")
