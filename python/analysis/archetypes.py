"""Silent decks by archetype (jack, 2026-11-06: "the deck building need to build towards a certain archetype not just have the average number of cards
the human has"). K-means on Heart winners' decks at the Act 2 boss found decks defined by one card or one style (notes/silent-training.md); `arch` labels a deck
by those rules, the defining card first, then what it holds most of. No torch, so system python will do.

    python3 python/analysis/archetypes.py --lines ~/sts-data/bosses-silent3-all/eval.jsonl.gz ~/sts-data/bosses-silent3-all/train.jsonl.gz
    python3 python/analysis/archetypes.py --dump python/runs/silent-cards-1/own
    python3 python/analysis/archetypes.py --lines ... --write-ids ~/sts-data/archetype-ids
    python3 python/analysis/archetypes.py --lines ... --write-pool ~/sts-data/late-balanced.jsonl.gz

--lines (human_bosses.py --all-outcomes) prints, by boss: each archetype's share of arrivals and of Heart wins, and what became of them. --dump (search_eval.py --decks, or
any summaries with "deck") prints the same for a network's decks at the end of its runs, past --min-floor. --write-ids writes each archetype's Heart winners' play ids (labelled by
their deck at the Act 2 boss) for human_dataset.py --play-ids. --write-pool writes lines balanced across archetypes for train_runs.py --late-starts: at most --cap lines of each
(boss, what became of the run, archetype), so that a run's value sees each archetype win and lose rather than winners' average.
"""
import argparse
import collections
import glob
import gzip
import hashlib
import json
import os

SHIV = {"Blade Dance", "Cloak And Dagger", "Infinite Blades", "Storm of Steel", "Accuracy", "A Thousand Cuts"}
POISON = {"Deadly Poison", "Poisoned Stab", "Bouncing Flask", "Noxious Fumes", "Catalyst", "Crippling Poison", "Corpse Explosion"}
DISCARD = {"Acrobatics", "Tactician", "Reflex", "Prepared", "Calculated Gamble", "Concentrate", "Tools of the Trade", "Eviscerate", "Underhanded Strike", "Unload", "Expertise"}
ATTACK = {"Dash", "Backstab", "Die Die Die", "Glass Knife", "Predator", "Skewer", "Masterful Stab", "Quick Slash", "Slice", "Dagger Spray", "Dagger Throw", "Flying Knee", "Sucker Punch",
          "Heel Hook", "All Out Attack", "Riddle With Holes", "Finisher", "Endless Agony", "Choke", "Flechettes", "Bane"}
ORDER = ["grand finale", "apparitions", "wraith form", "discard", "poison", "shiv", "attacks/mixed"]
# The game's names, where the human records' ids differ.
JAVA_ID = {"Cloak and Dagger": "Cloak And Dagger", "Wraith Form": "Wraith Form v2", "Apparition": "Ghostly", "Crippling Cloud": "Crippling Poison", "Sneaky Strike": "Underhanded Strike",
           "All-Out Attack": "All Out Attack", "Well-Laid Plans": "Well Laid Plans", "Piercing Wail": "PiercingWail"}


def arch(cards):
    """The archetype of a deck: its cards' ids or names ("+" for upgraded is fine)."""
    n = collections.Counter(JAVA_ID.get(c.rstrip("+"), c.rstrip("+")) for c in cards)
    if n["Grand Finale"]:
        return "grand finale"
    if n["Ghostly"]:
        return "apparitions"
    if n["Wraith Form v2"]:
        return "wraith form"
    held = lambda group: sum(n[c] for c in group)  # noqa: E731
    poison, shiv, discard = held(POISON), held(SHIV), held(DISCARD)
    if discard >= 3 and discard >= poison:
        return "discard"
    if poison >= 3 and poison >= shiv:
        return "poison"
    if shiv >= 3:
        return "shiv"
    return "attacks/mixed"


def read_lines(paths):
    for path in paths:
        with gzip.open(path, "rt") as f:
            for line in f:
                yield line, json.loads(line)


def lines_table(paths):
    by_floor = collections.defaultdict(list)
    for _, r in read_lines(paths):
        by_floor[r["floor"]].append((arch(c[0] for c in r["deck"]), r["outcome"]["label"], len(r["deck"]), len(r["relics"]), r["hp"] / max(r["max_hp"], 1)))
    for floor, rows in sorted(by_floor.items(), reverse=True):
        hearts = sum(x[1] == "heart" for x in rows)
        print(f"Arrivals at floor {floor}: {len(rows)}, of them the Heart beaten {hearts / len(rows):.1%}")
        for a in ORDER:
            g = [x for x in rows if x[0] == a]
            if g:
                print(f"  {a:14} {len(g) / len(rows):6.1%} of arrivals {sum(x[1] == 'heart' for x in g) / max(hearts, 1):6.1%} of Heart wins  from here: the pair {sum(x[1] in ('heart', 'won') for x in g) / len(g):6.1%}"
                      f"  the Heart {sum(x[1] == 'heart' for x in g) / len(g):6.1%}  deck {sum(x[2] for x in g) / len(g):4.1f} relics {sum(x[3] for x in g) / len(g):4.1f} HP {sum(x[4] for x in g) / len(g):3.0%}")


def dump_table(prefixes, min_floor):
    for prefix in prefixes:
        rows = []
        for path in glob.glob(f"{prefix}.*.jsonl") + glob.glob(f"{prefix}.jsonl"):
            with open(path) as f:
                rows += [json.loads(line) for line in f]
        late = [r for r in rows if r["floor"] >= min_floor and "deck" in r]
        print(f"{prefix}: {len(rows)} runs, {len(late)} past floor {min_floor - 1} with a deck, at the end of the run")
        for a in ORDER:
            g = [r for r in late if arch(r["deck"]) == a]
            if g:
                print(f"  {a:14} {len(g) / len(late):6.1%} {len(g):5} runs  Act 3 {sum(r['act'] >= 3 for r in g) / len(g):6.1%}  the pair {sum(r['won'] or r['act'] >= 4 for r in g) / len(g):6.1%}"
                      f"  the Heart {sum(r['won'] and r['act'] >= 4 for r in g) / len(g):6.1%}  floor {sum(r['floor'] for r in g) / len(g):4.1f}  deck {sum(len(r['deck']) for r in g) / len(g):4.1f}")


def write_ids(paths, out, floor=33):
    os.makedirs(out, exist_ok=True)
    ids = collections.defaultdict(list)
    for _, r in read_lines(paths):
        if r["floor"] == floor and r["outcome"]["label"] == "heart":
            ids[arch(c[0] for c in r["deck"])].append(r["play_id"])
    for a, group in ids.items():
        with open(os.path.join(out, a.replace("/", "-").replace(" ", "-") + ".txt"), "w") as f:
            f.write("\n".join(group) + "\n")
        print(f"{a}: {len(group)} Heart winners")


def write_pool(paths, out, cap):
    cells = collections.defaultdict(list)
    for line, r in read_lines(paths):
        key = (r["floor"], r["outcome"]["label"], r["outcome"].get("died_act"), arch(c[0] for c in r["deck"]))
        cells[key].append((hashlib.md5(r["play_id"].encode()).hexdigest(), line))
    n = 0
    with gzip.open(out, "wt") as f:
        for key, group in sorted(cells.items(), key=str):
            for _, line in sorted(group)[:cap]:
                f.write(line)
                n += 1
    print(f"{n} lines in {out}, from {sum(len(g) for g in cells.values())}; at most {cap} of each boss, ending and archetype")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lines", nargs="*", default=None, help="human_bosses.py --all-outcomes files")
    p.add_argument("--dump", nargs="*", default=None, help="prefixes of run summaries holding decks")
    p.add_argument("--min-floor", type=int, default=31, help="--dump: only runs that got this far")
    p.add_argument("--write-ids", default=None, help="a folder: <archetype>.txt, its Heart winners' play ids")
    p.add_argument("--write-pool", default=None, help="a .jsonl.gz of lines balanced across archetypes")
    p.add_argument("--cap", type=int, default=350, help="--write-pool: lines at most of each boss, ending and archetype")
    args = p.parse_args()
    if args.dump:
        dump_table(args.dump, args.min_floor)
    if args.lines:
        if args.write_ids:
            write_ids(args.lines, args.write_ids)
        elif args.write_pool:
            write_pool(args.lines, args.write_pool, args.cap)
        else:
            lines_table(args.lines)


if __name__ == "__main__":
    main()
