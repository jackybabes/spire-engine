"""Boss fights from human runs, as practice scenarios: the deck, relics, potions and HP a winning
player brought to a boss, and which boss(es), for `train_runs.py --scenarios`.

    .venv/bin/python python/human/human_bosses.py ~/sts-data/ironclad/runs --out ~/sts-data/bosses [--min-asc 20]
    .venv/bin/python python/human/human_bosses.py ~/sts-data/silent/runs --out ~/sts-data/bosses-silent --character THE_SILENT
    .venv/bin/python python/human/human_bosses.py ~/sts-data/defect/runs --out ~/sts-data/bosses-defect --character DEFECT \
        --min-asc 0 --kinds boss1,boss2,boss3,boss4
    .venv/bin/python python/human/human_bosses.py ~/sts-data/silent3/runs --out ~/sts-data/bosses-silent3-all --character THE_SILENT \
        --kinds boss1,boss2 --all-outcomes --workers 4

From each kept run of the character (`human_runs.keep`) at --min-asc or above that won, a scenario
for each fight of the kinds in --kinds (default boss2,boss3):
- boss1, boss2: the Act 1 and Act 2 bosses (floors 16 and 33);
- boss3: the Act 3 boss, which at A20 is two bosses back to back (floors 50 and 51);
- boss4: the Heart (floor 55, found by name: only runs that took the keys reach it, and a run
  that won it is a won run);
- elite1, elite2, elite3: every elite fight in that act (`"room": "elite"`).
The state is the run replayed (`human_runs.replay`) up to the boss floor: cards and relics from
the recorded changes before it, plus the unrecorded cards `human_runs.rebuild` fills in from the
final deck; HP and max HP at the end of the floor before. Potions aren't recorded as held: those
obtained before the floor (rewards, shops, events) minus as many as were used before it, oldest
first, at most the slots the ascension and Potion Belt give. Writes train.jsonl and eval.jsonl
(1 run in 20, by play id) under --out.

--clean leaves out cheated and abandoned runs (`human_runs.dirty`), makes the unrecorded card changes where a boss relic
made them (`human_runs.late_fill`), so an Act 1 deck doesn't hold what a later Astrolabe gave, and gives each state the
relic Neow gave (`human_runs.start_relics`; on a sample a third of winners took one, a fifth of those who died in Act 2).
Data built without it, all of it before 2026-11-06, has none of these.

--all-outcomes (implies --clean; its own --out: not for `train_runs.py --scenarios`, whose fights are winners') also keeps the runs
that did not win, as placebos for `analysis/play_on.py` and the contrast in `train_runs.py --late-starts`. Every line says what became
of its run: `"outcome": {"label": "heart" (beat the Heart), "won"
(both Act 3 bosses, no further) or "died", "floor_reached": N, and if it died "died_act": A, "killed_by": ...}`, and
adds what a run playing on needs (`gold`, `purges`: shop removals so far). A run that died at a boss has that state too (the
game records the fight it died in); boss3 needs both A20 bosses fought, so it leaves out deaths at the first.

The dump is untrusted data: this only parses it.
"""
import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders
import human.human_runs as hr  # noqa: E402

BOSS_FLOORS = {1: [16], 2: [33], 3: [50, 51]}
# The dump's elite encounters by act (the engine's encounters::ACT1_ELITES...).
ELITES = {1: {"Gremlin Nob", "Lagavulin", "3 Sentries"}, 2: {"Gremlin Leader", "Slavers", "Book of Stabbing"},
          3: {"Giant Head", "Nemesis", "Reptomancer"}}
# Each act's bosses: a run whose boss floor held something else (a few dozen of the Defect's,
# their floors numbered differently) gives no scenario for it.
BOSSES = {1: {"Slime Boss", "The Guardian", "Hexaghost"}, 2: {"Collector", "Champ", "Automaton"},
          3: {"Donu and Deca", "Time Eater", "Awakened One"}, 4: {"The Heart"}}
# A run whose max HP ever passed this is left out: cheated or modded (2026-10-24: 690 of the
# Defect's fights had 250-1,000,050, most about a million; she starts on 75, fruit takes it to
# about 150).
MAX_HP = 250
KINDS = ("boss1", "boss2", "boss3", "boss4", "elite1", "elite2", "elite3")
LIST_FIELDS = [("card_choices", "floor"), ("relics_obtained", "floor"), ("campfire_choices", "floor"), ("event_choices", "floor")]


def before(run, floor):
    """A copy of the run with only the changes recorded before `floor`."""
    r = dict(run)
    for key, f in LIST_FIELDS:
        r[key] = [x for x in run.get(key, []) if int(x[f]) < floor]
    pairs = [(i, fl) for i, fl in zip(run.get("items_purchased", []), run.get("item_purchase_floors", [])) if int(fl) < floor]
    r["items_purchased"], r["item_purchase_floors"] = [p[0] for p in pairs], [p[1] for p in pairs]
    pairs = [(i, fl) for i, fl in zip(run.get("items_purged", []), run.get("items_purged_floors", [])) if int(fl) < floor]
    r["items_purged"], r["items_purged_floors"] = [p[0] for p in pairs], [p[1] for p in pairs]
    r["boss_relics"] = [b for i, b in enumerate(run.get("boss_relics", [])) if i < len(hr.CHEST_FLOORS) and hr.CHEST_FLOORS[i] < floor]
    return r


def is_potion(item):
    """A shop purchase that is neither a card nor a relic."""
    return hr.split(item)[0] not in (hr.CARD_IDS or ()) and item not in (hr.RELIC_IDS or ())


def potions_before(run, floor, relics):
    """Potions held when `floor` starts (see the module docs)."""
    gains = [(int(p["floor"]), p["key"]) for p in run.get("potions_obtained", [])]
    gains += [(int(fl), i) for i, fl in zip(run.get("items_purchased", []), run.get("item_purchase_floors", [])) if is_potion(i)]
    uses = [int(f) for f in run.get("potions_floor_usage", [])]
    slots = (2 if run.get("ascension_level", 0) >= 11 else 3) + (2 if "Potion Belt" in relics else 0)
    held = []
    events = sorted([(f, 0, k) for f, k in gains] + [(f, 1, None) for f in uses])
    for f, kind, key in events:
        if f >= floor:
            break
        if kind == 0:
            held.append(key)
            if len(held) > slots:
                held.pop(0)
        elif held:
            held.pop(0)
    return held[-slots:]


def outcome(run):
    """What became of the run (--all-outcomes; see the module docs)."""
    floor = int(run.get("floor_reached") or 0)
    if run.get("victory"):
        return {"label": "heart" if hr.heart_win(run) else "won", "floor_reached": floor}
    return {"label": "died", "floor_reached": floor, "died_act": 1 + min((floor - 1) // 17, 3), "killed_by": run.get("killed_by")}


def scenarios(run, min_asc, character="IRONCLAD", kinds=("boss2", "boss3"), clean=False, all_outcomes=False):
    if hr.keep(run, character=character) or not (all_outcomes or run.get("victory")) or run.get("ascension_level", 0) < min_asc:
        return []
    if clean and hr.dirty(run):
        return []
    enemies = {int(t["floor"]): t["enemies"] for t in run.get("damage_taken", []) if isinstance(t, dict) and "floor" in t and "enemies" in t}
    hp, max_hp = run.get("current_hp_per_floor", []), run.get("max_hp_per_floor", [])
    if any(isinstance(m, (int, float)) and m > MAX_HP for m in max_hp):
        return []
    try:
        fill, _ = hr.rebuild(run)
    except KeyError:
        return []
    # The fights: (act, room, floors), bosses and then elites.
    fights = []
    for act, floors in BOSS_FLOORS.items():
        if f"boss{act}" in kinds:
            fights.append((act, "boss", floors if run.get("ascension_level", 0) >= 20 else floors[:1]))
    if "boss4" in kinds:
        fights += [(4, "boss", [floor]) for floor, who in sorted(enemies.items()) if who == "The Heart"]
    path = run.get("path_per_floor", [])
    for floor, who in sorted(enemies.items()):
        act = 1 + (floor - 1) // 17
        if f"elite{act}" in kinds and 0 < floor <= len(path) and path[floor - 1] == "E" and who in ELITES.get(act, ()):
            fights.append((act, "elite", [floor]))
    out = []
    for act, room, floors in fights:
        if any(f not in enemies for f in floors) or len(hp) < floors[0] - 1 or floors[0] < 2:
            continue
        if room == "boss" and any(enemies[f] not in BOSSES[act] for f in floors):
            continue
        r = before(run, floors[0])
        deck, relics = hr.Deck(hr.start_deck(run)), hr.start_relics(run, clean)
        try:
            hr.replay(r, deck, relics, fill=fill, fill_floor=hr.late_fill(run) if clean else None)
        except KeyError:
            continue
        sc = {
            "play_id": run.get("play_id"), "character": character, "ascension": run.get("ascension_level", 0), "act": act, "floor": floors[0],
            "hp": hp[floors[0] - 2], "max_hp": max_hp[floors[0] - 2], "bosses": [enemies[f] for f in floors],
            "deck": deck.cards, "relics": relics, "potions": potions_before(run, floors[0], relics),
        }
        if room == "elite":
            sc["room"] = "elite"
        if all_outcomes:
            gold = run.get("gold_per_floor", [])
            sc.update(outcome=outcome(run), gold=gold[floors[0] - 2] if len(gold) >= floors[0] - 1 else None, purges=len(r["items_purged"]))
        out.append(sc)
    return out


def held_out(play_id):
    """Whether the run's fights go in eval.jsonl: 1 run in 20."""
    return int(hashlib.md5(str(play_id).encode()).hexdigest(), 16) % 20 == 0


def work(args):
    path, min_asc, character, kinds, clean, all_outcomes = args
    out, runs, bad, dirty = [], 0, 0, collections.Counter()
    with gzip.open(path, "rt") as f:
        for line in f:
            runs += 1
            try:
                run = json.loads(line)
                out.extend(scenarios(run, min_asc, character, kinds, clean, all_outcomes))
                if clean and run.get("ascension_level", 0) >= min_asc and not hr.keep(run, character=character) and hr.dirty(run):
                    dirty[hr.dirty(run)] += 1
            except (KeyError, ValueError, TypeError, IndexError, AttributeError):
                bad += 1  # a malformed run (2026-10-24: a fight without its enemies)
    return runs, bad, dirty, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", help="folder of human_extract.py's runs/*.jsonl.gz")
    p.add_argument("--out", required=True)
    p.add_argument("--min-asc", type=int, default=20)
    p.add_argument("--character", default="IRONCLAD", choices=sorted(hr.CHARACTERS), help="as the dump names it")
    p.add_argument("--kinds", default="boss2,boss3", help=f"fights to make scenarios of, of {','.join(KINDS)}")
    p.add_argument("--clean", action="store_true", help="no cheated or abandoned runs, cards in where boss relics gave them (see above)")
    p.add_argument("--all-outcomes", action="store_true", help="also the runs that did not win, labelled (see above; implies --clean)")
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    clean = a.clean or a.all_outcomes
    files = sorted(glob.glob(os.path.join(a.runs, "*.jsonl.gz")))
    os.makedirs(a.out, exist_ok=True)
    total, skipped, kept, labels, dirty = 0, 0, collections.Counter(), collections.Counter(), collections.Counter()
    with gzip.open(os.path.join(a.out, "train.jsonl.gz"), "wt") as tr, gzip.open(os.path.join(a.out, "eval.jsonl.gz"), "wt") as ev, Pool(a.workers) as pool:
        jobs = [(f, a.min_asc, a.character, tuple(a.kinds.split(",")), clean, a.all_outcomes) for f in files]
        for runs, bad, left_out, out in pool.imap_unordered(work, jobs, chunksize=4):
            total += runs
            skipped += bad
            dirty += left_out
            for s in out:
                out_of_training = held_out(s["play_id"])
                (ev if out_of_training else tr).write(json.dumps(s) + "\n")
                kept[(s.get("room", "boss") + str(s["act"]), "eval" if out_of_training else "train")] += 1
                if a.all_outcomes:
                    o = s["outcome"]
                    labels[(s.get("room", "boss") + str(s["act"]), o["label"] + str(o.get("died_act", "")))] += 1
    print(f"{total:,} runs read ({skipped} malformed, skipped); scenarios: {dict(kept)}")
    if clean:
        print(f"left out by --clean: {dict(dirty)}")
    if a.all_outcomes:
        print(f"by what became of the run: {dict(sorted(labels.items()))}")


if __name__ == "__main__":
    main()
