"""Checks for python/human/human_runs.py's replay (plain asserts): python python/tests/test_human_runs.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

from collections import Counter

import human.human_runs as hr

# A boss relic that replaces the starter takes its slot (the dump lists both in a run's relics).
deck, relics = hr.Deck(["Strike_P", "Defend_P"]), ["PureWater", "Anchor"]
hr.gain_relic(deck, relics, "HolyWater", Counter())
hr.gain_relic(deck, relics, "Vajra", Counter())
assert relics == ["HolyWater", "Anchor", "Vajra"], relics

# What looks cheated or wasn't played out (`keep` passes all of these).
played = {"victory": True, "gold_per_floor": [99, 120, 140, 1500], "relics": ["Vajra"] * 29, "item_purchase_floors": [5] * 7 + [30] * 5}
assert hr.dirty(played) is None
assert hr.dirty({**played, "gold_per_floor": [99, 1200, 140]}) == "1,000+ gold within 3 floors"
assert hr.dirty({**played, "relics": ["Vajra"] * 30}) == "30+ relics"
assert hr.dirty({**played, "item_purchase_floors": [5] * 8}) == "8+ things bought by floor 6"
assert hr.dirty({**played, "chose_seed": True}) == "its seed chosen"
assert hr.dirty({**played, "victory": False, "killed_by": None}) == "abandoned"
assert hr.dirty({**played, "victory": False, "killed_by": "Hexaghost"}) is None
assert hr.dirty({"victory": True}) is None

# Unexplained cards: at floor 0 by default; with `late_fill`, after the boss relic that changed the deck, even when only
# a copy of the run cut off before the chest is replayed.
start = hr.CHARACTERS["THE_SILENT"][0]
run = {"character_chosen": "THE_SILENT", "ascension_level": 0, "card_choices": [{"floor": 5, "picked": "Slice", "not_picked": []}], "master_deck": start[1:] + ["Slice", "Malaise"],
       "boss_relics": [{"picked": "Astrolabe"}]}
assert hr.late_fill(run) == 17
assert hr.late_fill({**run, "boss_relics": [{"picked": "Vajra"}, {"picked": "Empty Cage"}, {"picked": "Pandora's Box"}]}) == 34
assert hr.late_fill({**run, "boss_relics": [{"picked": "Vajra"}]}) is None and hr.late_fill({}) is None
fill, plain = hr.rebuild(run)
assert not plain and fill == (Counter(["Malaise"]), Counter(["Strike_G"]))
deck = hr.Deck(start)
hr.replay(run, deck, [], fill=fill)
assert deck.counts(False) == hr.Deck(run["master_deck"]).counts(False)
deck = hr.Deck(start)
hr.replay(run, deck, [], fill=fill, fill_floor=17)
assert deck.counts(False) == hr.Deck(run["master_deck"]).counts(False)
cut = {**run, "boss_relics": []}
deck = hr.Deck(start)
hr.replay(cut, deck, [], fill=fill)
assert "Malaise" in deck.counts(False)
deck = hr.Deck(start)
hr.replay(cut, deck, [], fill=fill, fill_floor=17)
assert deck.counts(False) == hr.Deck(start + ["Slice"]).counts(False)

# Decisions: the same choices, the state before them as asked. A choice before the chest doesn't see the late cards.
run2 = {**run, "card_choices": [{"floor": 5, "picked": "Slice", "not_picked": []}, {"floor": 20, "picked": "Dash", "not_picked": []}], "master_deck": run["master_deck"] + ["Dash"]}
default, clean = hr.decisions(run2), hr.decisions(run2, True)
assert [d["floor"] for d in default] == [d["floor"] for d in clean]
at = lambda decs, f, card: card in [d for d in decs if d["floor"] == f and d["kind"] == "card"][0]["deck"]  # noqa: E731
assert at(default, 5, "Malaise") and not at(clean, 5, "Malaise") and at(clean, 20, "Malaise")

# Neow's relic: no record names it, held from the start if asked. A boss relic takes the starter's place, and a Ring of
# the Serpent from a chest doesn't bring the starter back.
told = {"character_chosen": "THE_SILENT", "ascension_level": 0, "relics_obtained": [{"floor": 3, "key": "Anchor"}], "boss_relics": [{"picked": "Sozu"}]}
snake = hr.CHARACTERS["THE_SILENT"][1]
assert hr.start_relics({**told, "neow_bonus": "BOSS_RELIC", "relics": ["Cursed Key", "Anchor", "Sozu"]}) == [snake]
assert hr.start_relics({**told, "neow_bonus": "BOSS_RELIC", "relics": ["Cursed Key", "Anchor", "Sozu"]}, True) == ["Cursed Key"]
assert hr.start_relics({**told, "neow_bonus": "ONE_RARE_RELIC", "relics": [snake, "Girya", "Anchor", "Sozu"]}, True) == [snake, "Girya"]
assert hr.start_relics({**told, "neow_bonus": "RANDOM_COMMON_RELIC", "relics": [snake, "Vajra", "Anchor", "Sozu"]}, True) == [snake, "Vajra"]
# No relic from Neow, whatever else is unexplained; or an event took it.
assert hr.start_relics({**told, "neow_bonus": "THREE_ENEMY_KILL", "relics": [snake, "NeowsBlessing", "Anchor", "Sozu"]}, True) == [snake]
assert hr.start_relics({**told, "neow_bonus": "BOSS_RELIC", "relics": ["Anchor", "Sozu"]}, True) == [snake]
serpent = {**told, "neow_bonus": "BOSS_RELIC", "relics": ["Cursed Key", "Anchor", "Ring of the Serpent"], "boss_relics": [{"picked": "Ring of the Serpent"}]}
relics = hr.start_relics(serpent, True)
hr.replay(serpent, hr.Deck(start), relics)
assert relics == ["Cursed Key", "Anchor", "Ring of the Serpent"], relics

print("ok")
