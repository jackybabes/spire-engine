"""A fixed, balanced sample of human_bosses.py's scenarios, for an eval that runs often.

    .venv/bin/python python/human/sample_scenarios.py ~/sts-data/bosses-defect/eval.jsonl.gz \\
        ~/sts-data/bosses-defect/eval-512.jsonl.gz --per 32

Takes --per scenarios of each act in each ascension band (A0-5, A6-10, A11-15, A16-20), at random
with --seed (the held-out set has about 21,000 of the Defect's, too many to play every 10 updates
on the CPU), and writes them in act order. Fewer if a band has fewer.
"""
import argparse
import collections
import gzip
import json
import random


def band(a):
    return 0 if a <= 5 else 1 if a <= 10 else 2 if a <= 15 else 3


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenarios")
    p.add_argument("out")
    p.add_argument("--per", type=int, default=32, help="scenarios of each act in each ascension band")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    groups = collections.defaultdict(list)
    with gzip.open(a.scenarios, "rt") as f:
        for line in f:
            s = json.loads(line)
            groups[(s["act"], band(s["ascension"]))].append(line.strip())
    rng = random.Random(a.seed)
    out = []
    for key in sorted(groups):
        out += rng.sample(groups[key], min(a.per, len(groups[key])))
    with gzip.open(a.out, "wt") as f:
        f.write("\n".join(out) + "\n")
    print(f"{len(out)} scenarios: " + ", ".join(f"act {k[0]} band {k[1]} {min(a.per, len(v))}" for k, v in sorted(groups.items())))


if __name__ == "__main__":
    main()
