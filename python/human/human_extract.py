"""Parallel pass over Mega Crit's 2020 metrics dump: one character's runs, decisions and stats
(the Ironclad's unless --character).

    .venv/bin/python python/human/human_extract.py ~/sts-data/Monthly_2020_1* --out ~/sts-data/ironclad
    .venv/bin/python python/human/human_extract.py ~/sts-data/Monthly_2020_1* --out ~/sts-data/silent --character THE_SILENT
    .venv/bin/python python/human/human_extract.py ~/sts-data/Monthly_2020_1* --out ~/sts-data/silent-clean --character THE_SILENT --clean
    python3 python/human/human_extract.py --report ~/sts-data/ironclad

For each chunk of the dump (a gzipped JSON list of runs), in parallel:
- every Ironclad run, as it is, to <out>/runs/<chunk>.jsonl.gz (so later passes read only those);
- the decisions of the runs `human_runs.keep` keeps (build 2020-07-30, no daily, endless, trial
  or beta runs, only cards and relics the engine has) to <out>/decisions/<chunk>.jsonl.gz;
- counts for <out>/stats.json: runs by character and build, why runs were left out, and for the
  kept runs wins by ascension, how far runs got, what killed them, card rewards (offered, taken,
  skipped), boss relics, rest sites, Neow, final decks and relics, won and lost.
--clean also leaves cheated and abandoned runs out of the decisions and the counts (`human_runs.dirty`), makes the
unrecorded card changes where a boss relic made them (`human_runs.late_fill`), and has Neow's relic held from the start
(`human_runs.start_relics`). Data built without it, all of it before 2026-11-06, has none of these.

The dump is untrusted data: this only parses it.
"""
import argparse
import collections
import gzip
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders
import human.human_runs as hr  # noqa: E402

C = collections.Counter


def new_stats():
    return {
        "files": 0, "runs": 0, "character": C(), "ironclad_build": C(), "left_out": C(),
        "kept": 0, "won": 0, "asc_runs": C(), "asc_won": C(), "floor": C(), "killed_by": C(),
        "offered": C(), "taken": C(), "skip": 0, "card_rewards": 0, "singing_bowl": 0,
        "boss_offered": C(), "boss_taken": C(), "boss_skip": 0, "campfire": C(),
        "neow_bonus": C(), "neow_cost": C(), "deck_won": C(), "deck_all": C(), "relic_won": C(), "relic_all": C(),
        "deck_size_won": C(), "deck_size_all": C(), "playtime_won": 0, "decisions": C(),
        "purged": C(), "bought": C(),
    }


def merge(a, b):
    for k, v in b.items():
        a[k] = a[k] + v if not isinstance(v, C) else a[k] + v
    return a


def name(card):
    return hr.split(card)[0]


def chunk(args):
    path, out, character, clean = args
    base = os.path.basename(path).replace("#", "_").removesuffix(".json.gz")
    s = new_stats()
    s["files"] = 1
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    runs_f = gzip.open(os.path.join(out, "runs", base + ".jsonl.gz"), "wt")
    dec_f = gzip.open(os.path.join(out, "decisions", base + ".jsonl.gz"), "wt")
    for rec in data:
        run = rec.get("event", rec) if isinstance(rec, dict) else None
        if not isinstance(run, dict):
            continue
        s["runs"] += 1
        s["character"][str(run.get("character_chosen"))] += 1
        if run.get("character_chosen") != character:
            continue
        s["ironclad_build"][str(run.get("build_version"))] += 1
        runs_f.write(json.dumps(run) + "\n")
        try:
            why = hr.keep(run, character=character) or (clean and hr.dirty(run))
        except Exception as e:  # noqa: BLE001 - malformed runs are counted, not fatal
            why = f"malformed ({type(e).__name__})"
        if why:
            s["left_out"][why] += 1
            continue
        try:
            decs = list(hr.decisions(run, clean))
        except Exception as e:  # noqa: BLE001
            s["left_out"][f"replay failed ({type(e).__name__})"] += 1
            continue
        for d in decs:
            dec_f.write(json.dumps(d) + "\n")
            s["decisions"][d["kind"]] += 1
        s["kept"] += 1
        won = bool(run.get("victory"))
        asc = int(run.get("ascension_level") or 0)
        s["won"] += won
        s["asc_runs"][asc] += 1
        s["asc_won"][asc] += won
        s["floor"][int(run.get("floor_reached") or 0)] += 1
        if not won:
            s["killed_by"][str(run.get("killed_by"))] += 1
        for c in run.get("card_choices", []):
            s["card_rewards"] += 1
            for x in [c.get("picked"), *c.get("not_picked", [])]:
                if x and x not in ("SKIP", "Singing Bowl"):
                    s["offered"][name(x)] += 1
            p = c.get("picked")
            if p == "SKIP":
                s["skip"] += 1
            elif p == "Singing Bowl":
                s["singing_bowl"] += 1
            elif p:
                s["taken"][name(p)] += 1
        for b in run.get("boss_relics", []):
            for r in [b.get("picked"), *b.get("not_picked", [])]:
                if r and r != "SKIP":
                    s["boss_offered"][r] += 1
            if b.get("picked") and b.get("picked") != "SKIP":
                s["boss_taken"][b["picked"]] += 1
            else:
                s["boss_skip"] += 1
        for c in run.get("campfire_choices", []):
            s["campfire"][str(c.get("key"))] += 1
        s["neow_bonus"][str(run.get("neow_bonus"))] += 1
        s["neow_cost"][str(run.get("neow_cost"))] += 1
        deck = [name(c) for c in run.get("master_deck", [])]
        s["deck_all"].update(set(deck))
        s["relic_all"].update(set(run.get("relics", [])))
        s["deck_size_all"][len(deck)] += 1
        s["purged"].update(name(c) for c in run.get("items_purged", []))
        s["bought"].update(name(c) for c in run.get("items_purchased", []))
        if won:
            s["deck_won"].update(set(deck))
            s["relic_won"].update(set(run.get("relics", [])))
            s["deck_size_won"][len(deck)] += 1
            s["playtime_won"] += int(run.get("playtime") or 0)
    runs_f.close()
    dec_f.close()
    return s


def run_all(paths, out, procs, character="IRONCLAD", clean=False):
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    os.makedirs(os.path.join(out, "decisions"), exist_ok=True)
    total = new_stats()
    with Pool(procs) as pool:
        for i, s in enumerate(pool.imap_unordered(chunk, [(p, out, character, clean) for p in paths], chunksize=4), 1):
            merge(total, s)
            if i % 250 == 0 or i == len(paths):
                print(f"{i}/{len(paths)} chunks, {total['runs']} runs, {sum(total['ironclad_build'].values())} {character}, "
                      f"{total['kept']} kept", flush=True)
    total["extracted"] = character  # whose runs `ironclad_build` and the rest count (the key names are the first pass's)
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(total, f)
    report(total)


def top(counter, n=15):
    return ", ".join(f"{k} {v}" for k, v in C(counter).most_common(n))


def report(s):
    pct = lambda a, b: f"{100 * a / b:.1f}%" if b else "-"  # noqa: E731
    iron = sum(s["ironclad_build"].values())
    print(f"\n{s['files']} chunks, {s['runs']} runs; by character: {top(s['character'], 6)}")
    print(f"{s.get('extracted', 'IRONCLAD')}: {iron} runs; builds: {top(s['ironclad_build'], 5)}")
    print(f"left out: {top(s['left_out'], 8)}")
    k = s["kept"]
    print(f"kept: {k} runs, won {s['won']} ({pct(s['won'], k)})")
    print("by ascension (runs, won):")
    for a in sorted(s["asc_runs"], key=int):
        print(f"  A{int(a):2}: {s['asc_runs'][a]:8}  {pct(s['asc_won'][a], s['asc_runs'][a]):>6}")
    fl = {int(f): n for f, n in s["floor"].items()}
    reach = lambda f: sum(n for x, n in fl.items() if x >= f)  # noqa: E731
    print(f"reached Act 2 {pct(reach(17), k)}, Act 3 {pct(reach(34), k)}, Act 4 {pct(reach(52), k)}; "
          f"died in Act 1 {pct(k - reach(17), k)}")
    print(f"killed by: {top(s['killed_by'], 12)}")
    n = s["card_rewards"]
    print(f"card rewards: {n}, skipped {pct(s['skip'], n)}, Singing Bowl {pct(s['singing_bowl'], n)}")
    rate = {c: s["taken"].get(c, 0) / o for c, o in s["offered"].items() if o >= 2000}
    best = sorted(rate, key=rate.get, reverse=True)
    print("most often taken when offered: " + ", ".join(f"{c} {rate[c]:.0%}" for c in best[:15]))
    print("least often taken: " + ", ".join(f"{c} {rate[c]:.0%}" for c in best[-12:]))
    brate = {r: s["boss_taken"].get(r, 0) / o for r, o in s["boss_offered"].items() if o >= 500}
    b = sorted(brate, key=brate.get, reverse=True)
    print("boss relics taken when offered: " + ", ".join(f"{r} {brate[r]:.0%}" for r in b[:10]) + " ... "
          + ", ".join(f"{r} {brate[r]:.0%}" for r in b[-5:]) + f"; skipped {s['boss_skip']}")
    camp = s["campfire"]
    print(f"rest sites: {', '.join(f'{c} {pct(v, sum(camp.values()))}' for c, v in C(camp).most_common(7))}")
    print(f"Neow: {top(s['neow_bonus'], 8)}")
    print(f"removed most: {top(s['purged'], 10)}")
    w = s["won"]
    lift = {c: (s["deck_won"].get(c, 0) / w) / (v / k) for c, v in s["deck_all"].items() if v >= 2000 and w}
    lb = sorted(lift, key=lift.get, reverse=True)
    print("cards in won decks vs all decks (ratio): " + ", ".join(f"{c} {lift[c]:.2f}" for c in lb[:12]))
    print("  ...rarer in won decks: " + ", ".join(f"{c} {lift[c]:.2f}" for c in lb[-8:]))
    rl = {r: (s["relic_won"].get(r, 0) / w) / (v / k) for r, v in s["relic_all"].items() if v >= 2000 and w}
    rb = sorted(rl, key=rl.get, reverse=True)
    print("relics in won runs vs all (ratio): " + ", ".join(f"{r} {rl[r]:.2f}" for r in rb[:12]))
    avg = lambda h: sum(int(x) * n for x, n in h.items()) / max(sum(h.values()), 1)  # noqa: E731
    print(f"final deck size: won {avg(s['deck_size_won']):.1f}, all {avg(s['deck_size_all']):.1f}; "
          f"average winning run {s['playtime_won'] / max(w, 1) / 60:.0f} min")
    print(f"decisions: {dict(s['decisions'])} ({sum(s['decisions'].values())} in all)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="chunk files or folders of them")
    p.add_argument("--out", default="ironclad")
    p.add_argument("--procs", type=int, default=max(os.cpu_count() - 2, 1))
    p.add_argument("--report", action="store_true", help="print <paths[0]>/stats.json")
    p.add_argument("--character", default="IRONCLAD", choices=sorted(hr.CHARACTERS), help="as the dump names it")
    p.add_argument("--clean", action="store_true", help="no cheated or abandoned runs, cards in where boss relics gave them (see above)")
    a = p.parse_args()
    if a.report:
        with open(os.path.join(a.paths[0], "stats.json")) as f:
            return report({k: (C(v) if isinstance(v, dict) else v) for k, v in json.load(f).items()})
    files = []
    for p_ in a.paths:
        files += sorted(os.path.join(p_, f) for f in os.listdir(p_) if f.endswith(".json.gz")) if os.path.isdir(p_) else [p_]
    run_all(files, a.out, a.procs, a.character, a.clean)


if __name__ == "__main__":
    sys.exit(main())
