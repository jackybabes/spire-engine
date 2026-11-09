"""Where runs end, from search/search_eval.py --dump's run summaries; with two dumps of the same seeds, which seeds each
won that the other lost and where the loser died.

    .venv/bin/python python/analysis/run_deaths.py <dump prefix> [<other dump prefix>] [--top 12]

A dump prefix names the files <prefix>.<process>.jsonl. A win is the Act 3 boss beaten (act 4 reached, or the run won).
"""
import argparse
import collections
import glob
import json

ELITES = {"Gremlin Nob", "Lagavulin", "3 Sentries", "Gremlin Leader", "Slavers", "Book of Stabbing", "Giant Head", "Nemesis",
          "Reptomancer"}
BOSSES = {"The Guardian", "Hexaghost", "Slime Boss", "Automaton", "Collector", "Champ", "Awakened One", "Time Eater",
          "Donu and Deca"}


def read(prefix):
    """The dump's runs by seed."""
    runs = {}
    for path in sorted(glob.glob(f"{prefix}.*.jsonl")):
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                runs[e["seed"]] = e
    return runs


def won(e):
    return e["won"] or e["act"] >= 4


def kind(e):
    """Where a lost run ended: "<act> boss / elite / hallway" (a hallway, event or other fight), or "timed out"."""
    if e.get("timed_out"):
        return "timed out"
    room = "boss" if e["died_in"] in BOSSES else "elite" if e["died_in"] in ELITES else "hallway"
    return f"act {e['act']} {room}"


def summary(runs, top):
    n = len(runs)
    lost = [e for e in runs.values() if not won(e)]
    print(f"{n} runs, Act 3 boss beaten {sum(won(e) for e in runs.values()) / n:.1%}, floor {sum(e['floor'] for e in runs.values()) / n:.1f}")
    for k, c in sorted(collections.Counter(kind(e) for e in lost).items()):
        held = [e["potions"] for e in lost if kind(e) == k and "potions" in e]
        print(f"  died {k:16s} {c:4d}  {c / n:5.1%} of runs" + (f"  potions held {sum(held) / len(held):.2f}" if held else ""))
    print("  deadliest:", ", ".join(f"{k} {c}" for k, c in collections.Counter(e["died_in"] for e in lost).most_common(top)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dumps", nargs="+")
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args()
    sets = [read(d) for d in args.dumps]
    for d, runs in zip(args.dumps, sets):
        print(f"== {d}")
        summary(runs, args.top)
    if len(sets) == 2:
        a, b = sets
        seeds = sorted(set(a) & set(b))
        only_a = [s for s in seeds if won(a[s]) and not won(b[s])]
        only_b = [s for s in seeds if won(b[s]) and not won(a[s])]
        print(f"== {len(seeds)} shared seeds: both won {sum(won(a[s]) and won(b[s]) for s in seeds)}, only the first {len(only_a)}, "
              f"only the second {len(only_b)}")
        for name, mine, other in (("first", only_a, b), ("second", only_b, a)):
            c = collections.Counter(kind(other[s]) for s in mine)
            print(f"  won by the {name} only, where the other died: " + ", ".join(f"{k} {n}" for k, n in sorted(c.items())))


if __name__ == "__main__":
    main()
