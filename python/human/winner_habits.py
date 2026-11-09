"""Human winners' habits between fights, for setting the run network's habit rewards (train_runs.py's
--rest-penalty, --rest-from, --smith-bonus, --deck-weight, --deck-free, --elite-bonus,
--hallway-potion-penalty).

    python3 python/human/winner_habits.py [~/sts-data/watcher] [--asc 20] [--all-wins]

Reads a character's runs (`human_extract.py`'s <out>/runs) at one ascension and, for the runs that
beat the Heart (won at floor 55 or more; --all-wins: any win), prints per run: rest sites and what
was done there, with the rest share by HP on arrival (the HP after the floor before); elites
fought; potions drunk in hallway fights; cards removed (shops and events); card rewards skipped;
and the deck's size at the end (quartiles). Chose the Watcher's habit rewards (2026-10-27).
"""
import argparse
import glob
import gzip
import json
import os
from collections import Counter

HEART_FLOOR = 55
HP_BUCKETS = (0.3, 0.45, 0.6, 0.75, 0.9, 1.01)


def quartiles(xs):
    xs = sorted(xs)
    return f"{xs[len(xs) // 4]} / {xs[len(xs) // 2]} / {xs[3 * len(xs) // 4]}" if xs else "-"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", nargs="?", default="~/sts-data/watcher")
    p.add_argument("--asc", type=int, default=20)
    p.add_argument("--all-wins", action="store_true")
    args = p.parse_args()

    n = 0
    camp = Counter()                       # what was done at rest sites
    by_hp = {b: Counter() for b in HP_BUCKETS}  # REST / SMITH / other, by HP on arrival
    per_run = {k: [] for k in ("rest sites", "rests", "smiths", "elites", "hallway potions", "removed", "skipped rewards", "deck")}
    rewards = skipped = 0
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(args.data), "runs", "*.jsonl.gz"))):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            if r.get("ascension_level") != args.asc or not r.get("victory"):
                continue
            if not args.all_wins and r.get("floor_reached", 0) < HEART_FLOOR:
                continue
            n += 1
            hp, mx = r.get("current_hp_per_floor", []), r.get("max_hp_per_floor", [])
            choices = r.get("campfire_choices", [])
            rests = smiths = 0
            for c in choices:
                key = c.get("key")
                camp[key] += 1
                rests += key == "REST"
                smiths += key == "SMITH"
                i = int(c.get("floor", 0)) - 2  # the floor before the rest site: HP on arrival
                if 0 <= i < len(hp) and i < len(mx) and mx[i]:
                    frac = min(hp[i] / mx[i], 1.0)  # (some records have HP past the max)
                    b = next(x for x in HP_BUCKETS if frac < x)
                    by_hp[b][key if key in ("REST", "SMITH") else "other"] += 1
            path = r.get("path_per_floor", [])
            used = Counter(int(x) for x in r.get("potions_floor_usage", []))
            per_run["rest sites"].append(len(choices))
            per_run["rests"].append(rests)
            per_run["smiths"].append(smiths)
            per_run["elites"].append(sum(1 for x in path if x == "E"))
            per_run["hallway potions"].append(sum(k for fl, k in used.items() if 0 < fl <= len(path) and path[fl - 1] == "M"))
            per_run["removed"].append(len(r.get("items_purged", [])))
            per_run["deck"].append(len(r.get("master_deck", [])))
            cc = r.get("card_choices", [])
            rewards += len(cc)
            sk = sum(1 for c in cc if c.get("picked") in ("SKIP", None))
            skipped += sk
            per_run["skipped rewards"].append(sk)
    if not n:
        raise SystemExit("no winning runs")
    print(f"{n:,} {'wins' if args.all_wins else 'Heart wins'} at A{args.asc}")
    total = sum(camp.values())
    print("rest sites: " + ", ".join(f"{k} {v / total:.0%}" for k, v in camp.most_common()))
    lo = 0.0
    for b in HP_BUCKETS:
        c = by_hp[b]
        t = sum(c.values())
        if t:
            print(f"  HP {lo:.0%}-{min(b, 1.0):.0%} on arrival: {t:6,} rest sites, rest {c['REST'] / t:4.0%}, smith {c['SMITH'] / t:4.0%}")
        lo = b
    for k, xs in per_run.items():
        print(f"{k:16} a run: mean {sum(xs) / len(xs):5.1f}, quartiles {quartiles(xs)}")
    print(f"card rewards skipped: {skipped / max(rewards, 1):.0%} of {rewards:,}")


if __name__ == "__main__":
    main()
