"""Human event choices from one character's saved runs (`human_extract.py`'s <out>/runs), without
another pass over the whole dump.

    .venv/bin/python python/human/event_decisions.py ~/sts-data/defect --out ~/sts-data/defect-events
    .venv/bin/python python/human/human_dataset.py ~/sts-data/defect-events/decisions --kinds event --out ~/sts-data/human-defect-events

Reads <src>/runs/*.jsonl.gz (one run a line) in parallel, keeps the runs `human_runs.keep` keeps
(won or not: human_dataset.py keeps the won ones), and writes their event decisions
(`human_runs.decisions`, kind "event") to <out>/decisions/<chunk>.jsonl.gz, with a count of the
events and choices. The runs are untrusted data: this only parses them.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import collections
import gzip
import json
from multiprocessing import Pool

import human.human_runs as hr


def chunk(args):
    path, out = args
    choices, left_out, n = collections.Counter(), collections.Counter(), 0
    with gzip.open(path, "rt") as f, gzip.open(os.path.join(out, "decisions", os.path.basename(path)), "wt") as dst:
        for line in f:
            try:
                run = json.loads(line)
                why = hr.keep(run, character=run.get("character_chosen"))
            except Exception as e:  # noqa: BLE001 - malformed runs are counted, not fatal
                why = f"malformed ({type(e).__name__})"
            if why:
                left_out[why] += 1
                continue
            try:
                decs = [d for d in hr.decisions(run) if d["kind"] == "event"]
            except Exception as e:  # noqa: BLE001
                left_out[f"replay failed ({type(e).__name__})"] += 1
                continue
            for d in decs:
                dst.write(json.dumps(d) + "\n")
                choices[(d["event"], d["taken"])] += 1
                n += 1
    return n, choices, left_out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="human_extract.py's --out for one character (with runs/)")
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=max(os.cpu_count() - 2, 1))
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, "decisions"), exist_ok=True)
    runs = os.path.join(a.src, "runs")
    paths = sorted(os.path.join(runs, f) for f in os.listdir(runs) if f.endswith(".jsonl.gz"))
    total, choices, left_out = 0, collections.Counter(), collections.Counter()
    with Pool(a.procs) as pool:
        for i, (n, c, lo) in enumerate(pool.imap_unordered(chunk, [(p_, a.out) for p_ in paths], chunksize=8), 1):
            total, choices, left_out = total + n, choices + c, left_out + lo
            if i % 500 == 0 or i == len(paths):
                print(f"{i}/{len(paths)} chunks, {total} event decisions", flush=True)
    print("left out:", dict(left_out.most_common(6)))
    by_event = collections.defaultdict(collections.Counter)
    for (ev, taken), c in choices.items():
        by_event[ev][taken] += c
    for ev, c in sorted(by_event.items(), key=lambda x: -sum(x[1].values())):
        print(f"  {ev} ({sum(c.values())}): " + ", ".join(f"{k} {v}" for k, v in c.most_common(6)))
    with open(os.path.join(a.out, "counts.json"), "w") as f:
        json.dump({"decisions": total, "left_out": left_out, "choices": {f"{e} / {t}": c for (e, t), c in choices.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
