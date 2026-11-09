"""How often a run checkpoint takes each card when offered, against human winners' rate.

    .venv/bin/python python/analysis/card_picks.py <run checkpoint> [--runs 256] [--ascension 20] [--seed-offset 0] \
        [--human ~/sts-data/defect/decisions] [--human-min-asc 15] [--human-heart] [--shards 1000] [--cards A,B] [--buckets]

Plays --runs runs greedily on the fixed eval seeds (a deck network's checkpoint as search_eval.py plays it: core.split.SplitPolicy,
its fight network fighting, its lookup if it trained with one) and counts, at every card reward screen, each card offered
and whether the network took it (or skipped the screen; leaving a reward unopened counts as a skip, as in a human record);
then reads the human decision shards (`human_extract.py`'s <out>/decisions, won runs at --human-min-asc or above, with
--human-heart the runs that beat the Heart, the first --shards files, leaving out runs that look cheated) for the same counts.
Prints the cards offered often enough to compare, the ones the network takes far less (and far more) often than the human winners
first, and the skip rates. Unlike a deck comparison it doesn't depend on how far the runs got. --buckets: the same again for each of search_eval.CARD_BUCKETS (floors
1-8, floors 9-16, Act 2, Act 3). --from-dump PREFIX: the counts from search_eval.py --dump's <PREFIX>.<process>.jsonl (with --picks or
--card-net) or one cell of a --cells --dump, <DUMP>.<cell>, instead of playing, which gives the comparison for those tests; no
checkpoint needed.

With --cells: do picks by a rule pay for this network? Plays the same runs once per cell, overriding only card rewards: a cell's
rules take a card if one is offered under them (the first rule's first, in the order given), otherwise the network's own choice
stands. Prints each cell's wins, Heart wins, Act 2, Act 3 and how many of those beat its bosses (search_eval.beat_act3: `won` misses a
run with the keys that died in Act 4), Act 4, floor, skips and how many picks it changed; against the cell with no rules (`own`), each
one's difference in Act 3 reach and in the Act 3 bosses on the same seeds. A rule is CONDITION[&CONDITION]:CARD,CARD, with the
cards as the game displays them (crates/sts-core/src/cards/data.rs: "Well-Laid Plans", "Cloak and Dagger") and CONDITIONS:
act1, act2, act3, act2+, act3+, noshiv (a deck with none of SHIV_MAKERS, read from the run's observation), fewshiv (at most one). A
cell is a name from CELLS (notes/silent-replan-2026-11-06.md, stage 2; --cells alone plays them all) or NAME=RULE;RULE:

    .venv/bin/python python/analysis/card_picks.py <run checkpoint> --cells own a b c d --runs 1024 --seed-offset 1024
    .venv/bin/python python/analysis/card_picks.py <run checkpoint> --cells own a "late=act2+&noshiv:Catalyst,Footwork" --seed-offset 2048

With --force: do the human winners' picks pay for this network? Plays the same runs four times,
overriding only card rewards: the network's own picks; taking any of --take when offered, in any act; never
taking --avoid (its next choice instead, a skip included); both. Prints each one's wins, Heart wins,
Act 3 and median floor. --take and --avoid default to the Defect's lists (TAKE, AVOID).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import collections
import glob
import gzip
import json

import numpy as np
import torch

import sts
from core.fight_model import pick_device
from core.lookup import Lookup, trained_events
from core.split import SplitPolicy
from core.sts_env import LAYOUT, RunEnv
from search.search_eval import CARD_BUCKETS, CardChoices, CardTally, beat_act3, cap_gpu, card_names, picks_lines, play
from training.train_runs import EVAL_SEED, load, load_fight_net, summarize

# Cards A15-20 human Defect winners take far more often (TAKE) and far less often (AVOID) than defect-runs-9
# at A20 (2026-10-26), for --force.
TAKE = ["Hologram", "Storm", "Reinforced Body", "Force Field", "Heatsinks", "Fission", "Creative AI"]
AVOID = ["Go for the Eyes", "Cold Snap", "Chill", "Hello World", "Ball Lightning"]
PICK, _ = LAYOUT["actions"]["pick"]
CARD_NAMES = set(sts.layout()["card_names"])

# The Silent's cards that make Shivs or pay for them (notes/silent-replan-2026-11-06.md).
SHIV_MAKERS = {"Blade Dance", "Accuracy", "Infinite Blades", "Cloak and Dagger", "Storm of Steel"}
CONDITIONS = {
    "act1": lambda view: view["act"] == 1,
    "act2": lambda view: view["act"] == 2,
    "act3": lambda view: view["act"] == 3,
    "act2+": lambda view: view["act"] >= 2,
    "act3+": lambda view: view["act"] >= 3,
    "noshiv": lambda view: not SHIV_MAKERS.intersection(card_names(view["deck"])),
    # At most one copy: the offer test's "shiv0-1" (balance.py), where Catalyst pays (Act 2, on the Heart: +2.41 ± 0.51, against
    # +0.31 ± 0.76 with two or more).
    "fewshiv": lambda view: sum(c in SHIV_MAKERS for c in card_names(view["deck"])) <= 1,
}
# The windows the offer test (2026-11-06: runs offered a card against runs not, on 160,624 A20 runs) supports.
POISON_ACT1 = "act1:Noxious Fumes,Bouncing Flask,Corpse Explosion"
CATALYST = "fewshiv:Catalyst"
LATE_SETUP = "act2+:Backflip,Acrobatics,Footwork"
CELLS = {"own": "", "a": POISON_ACT1, "b": CATALYST, "c": LATE_SETUP, "d": f"{POISON_ACT1};{CATALYST};{LATE_SETUP}"}


def load_policy(path, device, amp):
    """The checkpoint as it plays: its network, or a deck network's (train_runs.py --fight-net) SplitPolicy with its fight
    network fighting and its lookup if it trained with one. Returns (policy, lookup, checkpoint)."""
    net, ck = load(path, device)
    net.load_state_dict(ck["net"])
    net.eval()
    args = ck["args"]
    if not args.get("fight_net"):
        return net, None, ck
    fighter = load_fight_net(args["fight_net"], device)
    lookup = Lookup(fighter, device, amp, draws=args.get("lookup_draws", 4), every=args.get("lookup_every", False),
                    events=trained_events(ck["net"])) if args.get("lookup") else None
    return SplitPolicy(fighter, net), lookup, ck


def play_runs(policy, lookup, device, amp, a, character, rule=None, label="runs"):
    """The fixed runs (run j has seed EVAL_SEED + --seed-offset + j, as in search_eval.py), greedily, with `rule` at card rewards
    (CardChoices). Returns their summaries in run order, each with its "picks"."""
    b = max(min(a.batch, a.runs), 1)
    env = RunEnv(b, seed=EVAL_SEED + a.seed_offset, max_steps=8000, ascension=a.ascension, character=character)
    out, _ = play(policy, device, amp, env, lambda g, e: e * b + g < a.runs, None, 0.0, label, lookup=lookup, cards=CardChoices(b, rule=rule))
    return sorted(out, key=lambda e: e["seed"])


def parse_cell(text):
    """A --cells entry to (name, rules), each rule (conditions, cards). Raises ValueError on a condition or a card that doesn't
    exist: a misspelt card would silently be a rule that never takes anything."""
    name, named, body = text.partition("=")
    if not named:
        if name not in CELLS:
            raise ValueError(f"no cell {name!r} (cells: {', '.join(CELLS)}, or NAME=RULE;RULE)")
        body = CELLS[name]
    rules = []
    for rule in filter(None, body.split(";")):
        conditions, colon, cards = rule.partition(":")
        conditions, cards = conditions.split("&"), [c.strip() for c in cards.split(",") if c.strip()]
        unknown = [c for c in conditions if c not in CONDITIONS] + [c for c in cards if c not in CARD_NAMES]
        if not colon or not cards or unknown:
            raise ValueError(f"bad rule {rule!r} (CONDITION[&CONDITION]:CARD,CARD; unknown: {unknown}; conditions: {', '.join(CONDITIONS)})")
        rules.append((conditions, cards))
    return name, rules


def forced_slot(rules, view):
    """The action slot `rules` play at this card reward (`view`: the game's observation), or None."""
    cards = card_names(view["screen"]["cards"])
    for conditions, names in rules:
        if all(CONDITIONS[c](view) for c in conditions):
            for name in names:
                if name in cards:
                    return PICK + cards.index(name)
    return None


def blunt_rule(take, avoid):
    """--force: a card in `take` if offered (the first in `take`'s order), else the network's best choice that isn't in
    `avoid` (skipping counts)."""
    def rule(view, logits):
        cards = card_names(view["screen"]["cards"])
        wanted = [i for c in take for i, x in enumerate(cards) if x == c]
        if wanted:
            return PICK + wanted[0]
        if avoid:
            for a in torch.argsort(logits, descending=True).tolist():
                if logits[a] <= -1e8:  # (core.fight_model.masked)
                    break
                if not (PICK <= a < PICK + len(cards) and cards[a - PICK] in avoid):
                    return a
        return None
    return rule


def paired(values, base):
    """The mean difference of two same-seed lists of 0s and 1s and its standard error, in points."""
    d = np.array(values, dtype=float) - np.array(base, dtype=float)
    return f"{d.mean():+.1%} ± {d.std(ddof=1) / max(len(d), 1) ** 0.5:.1%}"


def cell_line(name, res, own):
    """A cell's line; `own`: the control's runs, on the same seeds."""
    r, t = summarize(res), CardTally().add_runs(res)
    beat, act3 = [beat_act3(e) for e in res], [e["act"] >= 3 for e in res]
    forced = collections.Counter(taken for e in res for _, _, taken, how in e["picks"] if how == "forced")
    line = (f"  {name:6} won {r['win_rate']:6.1%}  the Heart {r['win_heart']:5.1%}  Act 2 {r['reached_act2']:4.0%}  Act 3 {r['reached_act3']:4.0%} "
            f"(its bosses beaten {sum(beat) / max(sum(act3), 1):3.0%} of those)  Act 4 {r['reached_act4']:4.0%}  "
            f"floor {r['floor']:.1f}, median {np.median([e['floor'] for e in res]):.0f}  "
            f"card rewards {t.rewards[None] / max(len(res), 1):.1f} a run, skipped {t.skip_rate():.0%}")
    if forced:
        line += f"  changed {sum(forced.values()) / len(res):.2f} picks a run (" + ", ".join(f"{c} {n}" for c, n in forced.most_common()) + ")"
    if own is not None and own is not res:
        line += f"  against own: Act 3 {paired(act3, [e['act'] >= 3 for e in own])}, its bosses {paired(beat, [beat_act3(e) for e in own])}"
    return line


def cheated(r):
    """A decision record that gives its run away as cheated, as far as records can tell: 30+ relics, or 1,000+ gold within 3 floors
    (about 8.5% of the winners `human_runs.keep` passes; notes/silent-replan-2026-11-06.md, the new data filter)."""
    return len(r.get("relics") or ()) >= 30 or (r["floor"] <= 3 and (r.get("gold") or 0) > 1000)


def human_picks(path, min_asc, shards, heart=False):
    """The shards' card rewards as a CardTally, without the runs that look cheated (their number: its `left_out`)."""
    names = dict(zip(sts.layout()["card_java_ids"], sts.layout()["card_names"]))
    name = lambda s: names.get(s.split("+")[0], s.split("+")[0])  # noqa: E731
    tally = CardTally()
    tally.left_out = 0
    files = sorted(f for f in os.listdir(path) if f.endswith(".jsonl.gz"))[:shards]
    for f in files:
        with gzip.open(os.path.join(path, f), "rt") as fh:
            rows = [json.loads(line) for line in fh]
        bad = {r.get("play_id") for r in rows if cheated(r)}
        wanted = [r for r in rows if r.get("kind") == "card" and r.get("heart" if heart else "won") and int(r.get("ascension") or 0) >= min_asc]
        tally.left_out += len({r.get("play_id") for r in wanted} & bad)
        for r in wanted:
            if r.get("play_id") not in bad:
                skipped = r["taken"] in ("SKIP", "Singing Bowl")
                tally.add(r["floor"], [name(c) for c in r["offered"]], None if skipped else name(r["taken"]))
    return tally


def comparison(net, human, b, min_offered, cards):
    """Lines comparing two CardTallies in bucket `b` (None: every card reward)."""
    rows = []
    for c, n in net.offered[b].items():
        if n >= min_offered and human.offered[b][c] >= 30:
            rows.append((c, net.taken[b][c] / n, human.taken[b][c] / human.offered[b][c], n))
    rows.sort(key=lambda r: r[1] - r[2])
    fmt = lambda r: f"{r[0]} {r[1]:.0%} vs {r[2]:.0%} ({r[3]} offered)"  # noqa: E731
    lines = ["taken far less often than by human winners (network vs human, when offered): " + ", ".join(fmt(r) for r in rows[:15]),
             "taken far more often: " + ", ".join(fmt(r) for r in rows[::-1][:12])]
    for c in cards:
        if net.offered[b][c]:
            lines.append(f"  {c}: network {net.taken[b][c] / net.offered[b][c]:.0%} of {net.offered[b][c]} offers, "
                         f"human winners {human.taken[b][c] / max(human.offered[b][c], 1):.0%} of {human.offered[b][c]}")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", nargs="?", help="not needed with --from-dump")
    p.add_argument("--runs", type=int, default=256)
    p.add_argument("--batch", type=int, default=256, help="runs at once (search_eval.py's default)")
    p.add_argument("--ascension", type=int, default=20)
    p.add_argument("--seed-offset", type=int, default=0, help="run j has seed EVAL_SEED + this + j (e.g. 1024, 2048, 3072: seeds no checkpoint was chosen on)")
    p.add_argument("--device", default="auto")
    p.add_argument("--gpu-gb", type=float, default=None, help="GPU memory cap, GB (search_eval.py's)")
    p.add_argument("--human", default=os.path.expanduser("~/sts-data/defect/decisions"))
    p.add_argument("--human-min-asc", type=int, default=15)
    p.add_argument("--human-heart", action="store_true", help="the human runs that beat the Heart, not every won run")
    p.add_argument("--shards", type=int, default=1000)
    p.add_argument("--min-offered", type=int, default=30, help="cards offered fewer times than this (to the network) are left out")
    p.add_argument("--buckets", action="store_true", help="the rates by floor bucket too (with --cells: each cell's skips and most taken)")
    p.add_argument("--from-dump", default=None, help="the network's counts from <FROM_DUMP>.*.jsonl (search_eval.py --dump) or <FROM_DUMP>.jsonl (--cells --dump) instead of playing")
    p.add_argument("--dump", default=None, help="--cells: write each cell's runs to <DUMP>.<cell>.jsonl (each with its picks)")
    p.add_argument("--cells", nargs="*", default=None, help="the rule cells (see above) instead: names from CELLS or NAME=RULE;RULE; alone, every name in CELLS")
    p.add_argument("--force", action="store_true", help="the four forced-pick variants (--take, --avoid) instead")
    p.add_argument("--take", default=",".join(TAKE), help="--force: cards to take whenever offered, in order of preference (comma-separated)")
    p.add_argument("--avoid", default=",".join(AVOID), help="--force: cards never to take (comma-separated)")
    p.add_argument("--cards", default="Hologram,Capacitor,Defragment,Loop,Recursion,Storm,Electrodynamics,Echo Form,Go for the Eyes,Cold Snap",
                   help="cards whose rates to print whatever their difference (comma-separated)")
    a = p.parse_args()
    if not a.checkpoint and not a.from_dump:
        p.error("a checkpoint, or --from-dump")
    if a.from_dump and (a.cells is not None or a.force):
        p.error("--from-dump only reads counts")
    cells = None
    if a.cells is not None:
        try:
            cells = [parse_cell(c) for c in a.cells or CELLS]
        except ValueError as e:
            p.error(str(e))
        cells.sort(key=lambda c: bool(c[1]))  # the control first
    cards = [c for c in a.cards.split(",") if c]

    if a.from_dump:
        # One test's runs only: a process's file (a number), or a cell's whole name; <DUMP> alone would mix a --cells --dump's cells.
        out = [json.loads(line) for f in sorted(glob.glob(f"{a.from_dump}.[0-9]*.jsonl") or glob.glob(f"{a.from_dump}.jsonl")) for line in open(f)]
        if not out or "picks" not in out[0]:
            p.error(f"no picks in {a.from_dump}.<process>.jsonl or {a.from_dump}.jsonl (search_eval.py --picks or --card-net; for --cells --dump, name the cell: <DUMP>.<cell>)")
        print(f"{a.from_dump}: {len(out)} runs")
    else:
        device = pick_device(a.device)
        cap_gpu(device, a.gpu_gb)

        def amp():
            return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

        policy, lookup, ck = load_policy(a.checkpoint, device, amp)
        character = ck["args"].get("character", "ironclad")
        print(f"{a.checkpoint} (update {ck.get('update')}), A{a.ascension}, {a.runs} runs from seed offset {a.seed_offset}; "
              f"fights by {ck['args'].get('fight_net') or 'itself'}, lookup {bool(lookup)}, {character}", flush=True)
    if cells is not None:
        own = None
        for name, rules in cells:
            print(f"{name}: " + ("; ".join("&".join(c) + ": " + ", ".join(n) for c, n in rules) or "the network's own picks"), flush=True)
            res = play_runs(policy, lookup, device, amp, a, character, lambda view, logits: forced_slot(rules, view), name)
            if not rules:
                own = res
            print(cell_line(name, res, own), flush=True)
            if a.buckets:
                print("\n".join(picks_lines(res)), flush=True)
            if a.dump:
                with open(f"{a.dump}.{name}.jsonl", "w") as f:
                    for e in res:
                        f.write(json.dumps(e) + "\n")
        return
    if a.force:
        take, avoid = [c for c in a.take.split(",") if c], [c for c in a.avoid.split(",") if c]
        print(f"take {take}, avoid {avoid}", flush=True)
        for name, take, avoid in (("network's own", [], []), ("take", take, []), ("avoid", [], avoid), ("take and avoid", take, avoid)):
            res = play_runs(policy, lookup, device, amp, a, character, blunt_rule(take, avoid), name)
            won = np.mean([r["won"] for r in res])
            heart = np.mean([r["won"] and r["act"] >= 4 for r in res])
            print(f"  {name:16} won {won:6.1%}  the Heart {heart:5.1%}  Act 3 {np.mean([r['act'] >= 3 for r in res]):4.0%}  "
                  f"Act 4 {np.mean([r['act'] >= 4 for r in res]):4.0%}  floor median {np.median([r['floor'] for r in res]):.0f}, mean {np.mean([r['floor'] for r in res]):.1f}", flush=True)
        return
    if not a.from_dump:
        out = play_runs(policy, lookup, device, amp, a, character)
    net, human = CardTally().add_runs(out), human_picks(a.human, a.human_min_asc, a.shards, a.human_heart)
    winners = f"human {'Heart ' if a.human_heart else ''}winners A{a.human_min_asc}+"
    print(f"{net.rewards[None]} card rewards, skipped {net.skip_rate():.0%}; {winners}: {human.rewards[None]} card rewards, skipped {human.skip_rate():.0%} "
          f"({human.left_out} cheated runs left out)")
    print("\n".join(comparison(net, human, None, a.min_offered, cards)))
    if a.buckets:
        for b, name in enumerate(CARD_BUCKETS):
            print(f"{name}: {net.rewards[b]} card rewards, skipped {net.skip_rate(b):.0%}; {winners}: {human.rewards[b]}, skipped {human.skip_rate(b):.0%}")
            print("\n".join(comparison(net, human, b, a.min_offered, cards)))


if __name__ == "__main__":
    main()
