"""Human run histories (Mega Crit's 2020 metrics dump) turned into decisions to learn from.

    python3 python/human/human_runs.py stats   <chunk.json.gz ...>
    python3 python/human/human_runs.py extract <chunk.json.gz ...> --out decisions.jsonl.gz
    .venv/bin/python python/human/human_runs.py check <chunk.json.gz ...>     # encode, report what's kept

Each chunk of the dump is a gzipped JSON list of {"event": run}, one run per finished game, with
the fields the game's Metrics writes (card_choices, campfire_choices, boss_relics, event_choices,
items_purchased / items_purged with their floors, relics_obtained, master_deck, relics,
current_hp_per_floor, max_hp_per_floor, gold_per_floor, ...). Fights are not recorded.

`keep` picks the runs to use: one character's (the Ironclad by default; `CHARACTERS`), the given
build (default 2020-07-30, the last balance patch), not daily / endless / trial / beta, and only
cards and relics the engine has (other classes' cards from Prismatic Shard and the like are left
out). `dirty` says which of those are cheated or abandoned: nothing uses it unless asked to (data
built before 2026-11-06 has such runs in it).

`rebuild` replays a run's recorded changes from its character's starting deck: card rewards, shop
purchases and removals, rest-site upgrades and removals, and events' cards obtained, removed,
upgraded and transformed (a transform's result isn't recorded). What isn't recorded (Neow's
removals and transforms, random upgrades from relics and Apotheosis, transform results) is
filled in from the final `master_deck`: cards it has that the replay lacks are put in at floor 0,
cards the replay has that it lacks are taken out there. That gives an early deck later cards when
a boss relic made the changes: `late_fill` is the floor to make them on instead. `stats` reports
how often the replay alone already ends on the recorded deck.

`decisions` lists the choices with the state before each: the floor, act, HP, max HP, gold, room
type, the act's boss (if the run reached it), the ruby key, deck and relics, and what was offered
and taken. So far: card rewards (cards offered, the one taken or
a skip, or Singing Bowl), rest sites (rest, smith and the card, lift, toke and the card, dig,
recall), boss relics (offered and taken), shop card removals (the card removed) and events (the
event, the choice as the dump names it, and the relics it gave up; `crates/sts-py/src/human.rs`
maps the choice to the engine's options).

`encode` turns decision records into the run network's inputs (`sts.encode_human`): the arrays
`RunEnv.observe()` gives, for a run rebuilt in that state on that screen, and the action slot the
player took, for training the network to predict human choices. A rest site's smith or toke
gives two samples: the choice, then the card.

The dump is untrusted data: this only parses it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import collections
import gzip
import json
import sys

try:
    from core.sts_env import LAYOUT
    CARD_IDS = set(LAYOUT["card_java_ids"])
    RELIC_IDS = set(LAYOUT["relic_java_ids"])
except ImportError:  # without the sts module: no name checks
    CARD_IDS = RELIC_IDS = None

# Each character's starting deck, starting relic and basic Strike and Defend, by the dump's name
# (`character_chosen`); the engine's Character::starting_deck and starting_relic.
CHARACTERS = {
    "IRONCLAD": (["Strike_R"] * 5 + ["Defend_R"] * 4 + ["Bash"], "Burning Blood", ("Strike_R", "Defend_R")),
    "THE_SILENT": (["Strike_G"] * 5 + ["Defend_G"] * 5 + ["Survivor", "Neutralize"], "Ring of the Snake", ("Strike_G", "Defend_G")),
    "DEFECT": (["Strike_B"] * 4 + ["Defend_B"] * 4 + ["Zap", "Dualcast"], "Cracked Core", ("Strike_B", "Defend_B")),
    "WATCHER": (["Strike_P"] * 4 + ["Defend_P"] * 4 + ["Eruption", "Vigilance"], "PureWater", ("Strike_P", "Defend_P")),
}
START_DECK = CHARACTERS["IRONCLAD"][0]
CHEST_FLOORS = (17, 34, 51)  # boss_relics[i] is chosen on the floor after act i + 1's boss


def load(paths):
    for p in paths:
        with gzip.open(p, "rt") as f:
            for rec in json.load(f):
                yield rec.get("event", rec)


def split(card):
    """'Bash+1' -> ('Bash', 1)."""
    name, _, up = card.partition("+")
    return name, int(up) if up.isdigit() else 0


def character(run):
    """The run's character as the dump names it (the Ironclad if unknown)."""
    c = run.get("character_chosen")
    return c if c in CHARACTERS else "IRONCLAD"


def keep(run, build="2020-07-30", character="IRONCLAD"):
    """Why the run is left out, or None to use it."""
    if run.get("character_chosen") != character:
        return f"not {character}"
    if build and run.get("build_version") != build:
        return "other build"
    if run.get("is_daily") or run.get("is_endless") or run.get("is_trial") or run.get("is_beta"):
        return "daily, endless, trial or beta"
    if not run.get("card_choices") and run.get("floor_reached", 0) > 3:
        return "no card choices recorded"
    if CARD_IDS is not None:
        cards = {split(c)[0] for c in run.get("master_deck", [])}
        for c in run.get("card_choices", []):
            cards.update(split(x)[0] for x in [c["picked"], *c["not_picked"]] if x not in ("SKIP", "Singing Bowl"))
        if cards - CARD_IDS:
            return "cards the engine doesn't have"
        if set(run.get("relics", [])) - RELIC_IDS:
            return "relics the engine doesn't have"
    return None


def dirty(run):
    """Why a run `keep` passes looks cheated or wasn't played out, or None. About 8.5% of the Silent's A20 winners look cheated
    (notes/silent-replan-2026-11-06.md); abandons have no result to learn from."""
    if max((run.get("gold_per_floor") or [0])[:3]) > 1000:
        return "1,000+ gold within 3 floors"
    if len(run.get("relics", [])) >= 30:
        return "30+ relics"
    if sum(int(f) <= 6 for f in run.get("item_purchase_floors", [])) >= 8:
        return "8+ things bought by floor 6"
    if run.get("chose_seed"):
        return "its seed chosen"
    if not run.get("victory") and not run.get("killed_by"):
        return "abandoned"
    return None


class Deck:
    """A master deck as [card id, upgrades] pairs, in the order cards joined it."""

    def __init__(self, cards):
        self.cards = [list(split(c)) for c in cards]

    def add(self, card):
        self.cards.append(list(split(card)))

    def remove(self, card):
        """Removes a copy of `card` (its upgrade level if one is given, else the least upgraded)."""
        name, up = split(card)
        same = [c for c in self.cards if c[0] == name]
        if not same:
            return False
        exact = [c for c in same if c[1] == up] if "+" in card else []
        self.cards.remove(exact[0] if exact else min(same, key=lambda c: c[1]))
        return True

    def upgrade(self, card):
        name, _ = split(card)
        same = [c for c in self.cards if c[0] == name]
        if not same:
            return False
        c = min(same, key=lambda c: c[1])
        if c[1] == 0 or name == "Searing Blow":
            c[1] += 1
        return True

    def names(self):
        return [f"{n}+{u}" if u else n for n, u in self.cards]

    def counts(self, upgrades=True):
        return collections.Counter(self.names() if upgrades else [n for n, _ in self.cards])


def changes(run):
    """The run's recorded deck and relic changes, as (floor, order, kind, data), in play order:
    events, then rewards, then the shop, then the rest site on a floor."""
    out = []
    for e in run.get("event_choices", []):
        out.append((int(e["floor"]), 0, "event", e))
    for c in run.get("card_choices", []):
        out.append((int(c["floor"]), 1, "card", c))
    for r in run.get("relics_obtained", []):
        out.append((int(r["floor"]), 1, "relic", r["key"]))
    for item, floor in zip(run.get("items_purchased", []), run.get("item_purchase_floors", [])):
        out.append((int(floor), 2, "buy", item))
    for card, floor in zip(run.get("items_purged", []), run.get("items_purged_floors", [])):
        out.append((int(floor), 2, "purge", card))
    for c in run.get("campfire_choices", []):
        out.append((int(c["floor"]), 3, "campfire", c))
    for i, b in enumerate(run.get("boss_relics", [])):
        out.append((CHEST_FLOORS[i] if i < len(CHEST_FLOORS) else 99, 4, "boss", b))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


# Boss relics that change the deck when taken, unrecorded: the cards they add, and whether they
# turn every Strike and Defend into random cards (Pandora's Box).
RELIC_CARDS = {"Calling Bell": ["CurseOfTheBell"], "Necronomicon": ["Necronomicurse"]}
TRANSFORMS_BASICS = {"Pandora's Box"}
# Boss relics that take or change cards at the chest, unrecorded: Empty Cage, Astrolabe, Pandora's Box.
FILLS_LATE = {"Empty Cage", "Astrolabe", *TRANSFORMS_BASICS}


# Boss relics that replace the starter relic, taking its slot (the dump lists both in a run's
# relics; before 2026-10-27 the replay kept both too).
REPLACES_STARTER = {"Black Blood": "Burning Blood", "Ring of the Serpent": "Ring of the Snake", "FrozenCore": "Cracked Core",
                    "HolyWater": "PureWater"}


def gain_relic(deck, relics, relic, pending, basics=("Strike_R", "Defend_R")):
    starter = REPLACES_STARTER.get(relic)
    if starter in relics:
        relics[relics.index(starter)] = relic
    else:
        relics.append(relic)
    for c in RELIC_CARDS.get(relic, []):
        deck.add(c)
    if relic in TRANSFORMS_BASICS:
        for name, _ in [c for c in deck.cards if c[0] in basics]:
            deck.remove(name)
        # The random cards it gave are among the final deck's unexplained ones: in from here.
        for c in pending.elements():
            deck.add(c)
        pending.clear()


def late_fill(run):
    """The floor of the first boss relic that changed the deck unrecorded (`FILLS_LATE`), or None. With two, the first: a
    deck short of Pandora's cards until the next act's chest is further off than one early by an Astrolabe's three. Asks
    the whole run: a copy cut off before the chest (human_bosses.before) doesn't show it."""
    for floor, b in zip(CHEST_FLOORS, run.get("boss_relics", [])):
        if b.get("picked") in FILLS_LATE:
            return floor
    return None


def replay(run, deck, relics, on_decision=None, fill=None, fill_floor=None):
    """Applies the recorded changes to `deck` and `relics` (a list). With `fill` (missing,
    extra), the unrecorded changes are made too: removals at floor 0, and cards in at floor 0,
    or where Pandora's Box is taken if it is; with `fill_floor` (`late_fill`), all of them after
    that floor's boss relic instead, so not at all in a copy of the run cut off before it.
    `on_decision(kind, floor, data)` is called before each choice is applied."""
    pending = collections.Counter()
    basics = CHARACTERS[character(run)][2]

    def gain(r):
        gain_relic(deck, relics, r, pending, basics)

    if fill and fill_floor is None:
        missing, extra = fill
        for c in extra.elements():
            deck.remove(c)
        if any(b.get("picked") in TRANSFORMS_BASICS for b in run.get("boss_relics", [])):
            pending = collections.Counter(missing)
        else:
            for c in missing.elements():
                deck.add(c)
    for floor, _, kind, data in changes(run):
        if on_decision and kind in ("card", "campfire", "boss", "purge", "event"):
            on_decision(kind, floor, data)
        if kind == "card":
            if data["picked"] not in ("SKIP", "Singing Bowl"):
                deck.add(data["picked"])
        elif kind == "relic":
            gain(data)
        elif kind == "buy":
            if CARD_IDS is None or split(data)[0] in CARD_IDS:
                deck.add(data)
            elif data in (RELIC_IDS or ()):
                gain(data)
        elif kind == "purge":
            deck.remove(data)
        elif kind == "campfire":
            if data["key"] == "SMITH" and data.get("data"):
                deck.upgrade(data["data"])
            elif data["key"] == "PURGE" and data.get("data"):
                deck.remove(data["data"])
        elif kind == "boss":
            if data.get("picked") and data["picked"] != "SKIP":
                gain(data["picked"])
            if fill and floor == fill_floor:
                for c in fill[1].elements():
                    deck.remove(c)
                for c in fill[0].elements():
                    deck.add(c)
        elif kind == "event":
            for c in data.get("cards_removed", []) + data.get("cards_transformed", []):
                deck.remove(c)
            for c in data.get("cards_obtained", []):
                deck.add(c)
            for c in data.get("cards_upgraded", []):
                deck.upgrade(c)
            for r in data.get("relics_obtained", []):
                gain(r)
            for r in data.get("relics_lost", []):
                if r in relics:
                    relics.remove(r)


def start_deck(run):
    return CHARACTERS[character(run)][0] + (["AscendersBane"] if run.get("ascension_level", 0) >= 10 else [])


# Neow's blessings that give a relic. No record names it: it's the first relic in the final list that the recorded changes
# don't explain, a boss relic in the starter's place (so on 3,176 of 3,184 of the Silent's on a sample; most of the rest lost it to an event).
NEOW_RELICS = {"BOSS_RELIC", "ONE_RARE_RELIC", "RANDOM_COMMON_RELIC"}


def start_relics(run, neow=False):
    """The relics a run starts with: its starter, or with `neow` as Neow left them. (Data built without it, all of it before
    2026-11-06, lacks that relic and keeps the starter a boss relic took.)"""
    starter = CHARACTERS[character(run)][1]
    if not neow or run.get("neow_bonus") not in NEOW_RELICS:
        return [starter]
    replayed = [starter]
    replay(run, Deck(start_deck(run)), replayed)
    unexplained = [r for r in run.get("relics", []) if r not in replayed]
    if not unexplained:
        return [starter]
    return unexplained[:1] if run["neow_bonus"] == "BOSS_RELIC" else [starter, unexplained[0]]


def rebuild(run):
    """(fill, plain): the unrecorded cards in and out (by card, unupgraded: random upgrades are
    left out rather than guessed) that make the replay end on `master_deck`'s cards, and whether
    the replay alone already did."""
    deck = Deck(start_deck(run))
    replay(run, deck, [], fill=(collections.Counter(), collections.Counter()))
    final, have = Deck(run["master_deck"]).counts(False), deck.counts(False)
    return (final - have, have - final), final == have


# A win at floor 55 or later went on through the final act and beat the Heart (a run that stops at
# the Act 3 boss ends a victory on floor 51 at most).
HEART_FLOOR = 55


def heart_win(run):
    return bool(run.get("victory")) and int(run.get("floor_reached") or 0) >= HEART_FLOOR


def decisions(run, clean=False):
    """The run's choices, each with the state before it, and how the run ended (`floor_reached`,
    `heart`: whether it beat the Heart), the outcome a choice led to. `clean`: the unrecorded
    changes made where a boss relic made them (`late_fill`), and Neow's relic held from the start."""
    fill, _ = rebuild(run)
    deck, relics = Deck(start_deck(run)), start_relics(run, clean)
    hp, max_hp, gold = run.get("current_hp_per_floor", []), run.get("max_hp_per_floor", []), run.get("gold_per_floor", [])
    path = run.get("path_per_floor", [])
    # Each act's boss, from the fight on its boss floor (unknown if the run ended before it).
    bosses = {int(t["floor"]): t["enemies"] for t in run.get("damage_taken", []) if int(t["floor"]) in (16, 33, 50)}
    recalls = [int(c["floor"]) for c in run.get("campfire_choices", []) if c["key"] == "RECALL"]
    out = []

    def at(lst, floor):
        # The per-floor lists are indexed by floor - 1 (floor 0 is Neow: none recorded).
        return lst[floor - 1] if 0 < floor <= len(lst) else None

    def on_decision(kind, floor, data):
        act = 1 + min(max(floor - 1, 0) // 17, 3)
        # A rest site's or an event's choice is made on arrival: the HP of the floor before. The floor's
        # own is after it, a rest's heal included, which would show a rest at 40% HP as one at 70%.
        # An event's gold is the floor before's too (the floor's own has what the event gave or took).
        hp_floor = floor - 1 if kind in ("campfire", "event") else floor
        gold_floor = floor - 1 if kind == "event" else floor
        rec = {"play_id": run.get("play_id"), "character": character(run), "ascension": run.get("ascension_level", 0), "won": bool(run.get("victory")),
               "floor_reached": int(run.get("floor_reached") or 0), "heart": heart_win(run), "floor": floor, "act": act, "hp": at(hp, hp_floor), "max_hp": at(max_hp, hp_floor), "gold": at(gold, gold_floor),
               "room": at(path, floor), "boss": bosses.get(17 * act - 1), "ruby_key": any(f < floor for f in recalls),
               "deck": deck.names(), "relics": list(relics), "kind": kind}
        if kind == "card":
            rec["offered"] = [data["picked"], *data["not_picked"]] if data["picked"] not in ("SKIP", "Singing Bowl") else list(data["not_picked"])
            rec["taken"] = data["picked"]
        elif kind == "campfire":
            rec["taken"] = data["key"]
            rec["card"] = data.get("data")
        elif kind == "boss":
            rec["offered"] = [x for x in [data.get("picked"), *data.get("not_picked", [])] if x and x != "SKIP"]
            rec["taken"] = data.get("picked") or "SKIP"
        elif kind == "purge":
            rec["taken"] = data
        elif kind == "event":
            rec["event"] = data.get("event_name")
            rec["taken"] = data.get("player_choice")
            rec["relics_lost"] = data.get("relics_lost", [])
        out.append(rec)

    replay(run, deck, relics, on_decision=on_decision, fill=fill, fill_floor=late_fill(run) if clean else None)
    return out


def encode(records):
    """(obs, slots, source, skipped) for decision records: obs as RunEnv.observe() lays it out,
    the slot each sample's player took, the record each sample came from, and (record, reason)
    for records the rebuilt state can't express."""
    import numpy as np
    import sts
    from core.sts_env import ACTIONS, RUN_FLOAT_LEN, RUN_INT_LEN, RunEnv, _split
    ib, fb, mb, slots, source, skipped = sts.encode_human([json.dumps(r) for r in records])
    ints = np.frombuffer(ib, dtype="<i4").reshape(-1, RUN_INT_LEN).copy()
    floats = np.frombuffer(fb, dtype="<f4").reshape(-1, RUN_FLOAT_LEN).copy()
    mask = np.frombuffer(mb, dtype=np.uint8).reshape(-1, ACTIONS).astype(bool)
    obs = {"ints": ints, "floats": floats, "mask": mask}
    obs.update({"i_" + k: v for k, v in _split(ints, RunEnv.INTS).items()})
    obs.update({"f_" + k: v for k, v in _split(floats, RunEnv.FLOATS).items()})
    return obs, np.array(slots), np.array(source), skipped


def check(paths, build):
    records = [d for run in load(paths) if not keep(run, build) for d in decisions(run)]
    obs, slots, source, skipped = encode(records)
    kinds = collections.Counter(records[i]["kind"] for i in source)
    print(f"{len(records)} decisions -> {len(slots)} samples ({dict(kinds)}); {len(skipped)} records left out:")
    for why, n in collections.Counter(w for _, w in skipped).most_common(8):
        print(f"  {n:5}  {why}")
    import numpy as np
    assert obs["mask"][np.arange(len(slots)), slots].all()
    print("every sample's slot is allowed in its rebuilt state")


def stats(paths, build):
    reasons, kept, plain, upgrades_too, kinds = collections.Counter(), 0, 0, 0, collections.Counter()
    wins = 0
    for run in load(paths):
        why = keep(run, build)
        reasons[why or "kept"] += 1
        if why:
            continue
        kept += 1
        wins += bool(run.get("victory"))
        fill, ok = rebuild(run)
        plain += ok
        deck = Deck(start_deck(run))
        replay(run, deck, [], fill=fill)
        upgrades_too += deck.counts() == Deck(run["master_deck"]).counts()
        kinds.update(d["kind"] for d in decisions(run))
    print("runs:", dict(reasons.most_common()))
    if kept:
        print(f"kept {kept} ({wins} won); the recorded changes alone end on the recorded cards in {plain / kept:.0%}; "
              f"with the unrecorded ones filled in, upgrades match too in {upgrades_too / kept:.0%}")
        print("decisions:", dict(kinds), f"({sum(kinds.values()) / kept:.1f} per run)")


def extract(paths, build, out):
    n = 0
    with gzip.open(out, "wt") as f:
        for run in load(paths):
            if keep(run, build):
                continue
            for d in decisions(run):
                f.write(json.dumps(d) + "\n")
                n += 1
    print(f"{n} decisions -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["stats", "extract", "check"])
    p.add_argument("files", nargs="+")
    p.add_argument("--build", default="2020-07-30", help="game build to keep ('' for all)")
    p.add_argument("--out", default="decisions.jsonl.gz")
    args = p.parse_args()
    if args.mode == "stats":
        stats(args.files, args.build)
    elif args.mode == "check":
        check(args.files, args.build)
    else:
        extract(args.files, args.build, args.out)


if __name__ == "__main__":
    sys.exit(main())
