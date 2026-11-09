"""Can a fight network win with winners' decks? (jack, 2026-11-06: "the fight network needs to be able to win with those cards"; stage 3 of
notes/silent-replan-2026-11-06.md.) Greedy play from human boss states on as normal runs, and the same fighter's own arrivals beside them.

    .venv/bin/python python/analysis/play_on.py python/runs/silent-runs-5/update_1200.pt ~/sts-data/bosses-silent3-all/eval.jsonl.gz \
        ~/sts-data/bosses-silent3-all/train.jsonl.gz --fighter python/runs/silent-allstyles/update_600.pt --own ~/scripts/dumps/s1-allstyles

Lines come from `human_bosses.py --all-outcomes`. Each state is the run placed in that boss fight, with her deck, relics, potions, HP, gold and
removals (`Envs.set_scenarios(..., play_on=True)`), and it goes on as a normal run to a win or a death. The checkpoint makes the choices between
fights. Its fights are played by --fighter if given (on a plain run network too, unlike search_eval.py --fighter), else by its own --fight-net if it
has one, else by itself; the line it prints says which. A split checkpoint keeps its lookup, read with the fight network it trained with.

Groups (--groups; every fighter plays every group, with the same seeds):
    winners      won the run: both Act 3 bosses (`"label": "won"`), or the Heart too ("heart", also shown apart)
    finishers    other finishers, the first placebo: reached Act 3 or the final act and died there (strong decks too)
    act2-deaths  died in Act 2, the second placebo; at boss2 that's only those who died to that boss, the one way to die in Act 2 with its state
Also: heart, won, act1-deaths, act3-deaths, act4-deaths. --strip-relics leaves each state only its starter relic (her deck otherwise; the max
HP and upgrades relics gave stay): is it the cards, or the relics she had? --full-hp starts each state on full HP: is it the HP each brought (the
"brought HP" of its row; winners can arrive lower at the Act 1 boss, their risk being what their skill could carry)?

Per group: beat this boss, reached the next act, reached Act 3, both Act 3 bosses (the pair, "won" in the evals), the Heart, the floor, each
with its standard error; the keys it ended with; and what the state brought. --own <prefix>[,<prefix>...] reads search_eval.py --dumps of the same
fighter's own runs (<prefix>.<process>.jsonl: act, floor, died_in, won, keys) for the same rates from its own arrivals at that boss: every run whose
floor reached it. Those summaries hold nothing about the arrival itself (HP, deck). The reading: winners at twice its own and clearly above the
placebos means the deck is a real limit; all about equal means fights are. Twice its own can only show on the pair: its own next act is above 50% at
both bosses. The lines at the end of a kind read winners twice, all of them and those no training saw (the placebos' lines never were in training, unless it had late starts: below).

Keys: no state holds any (a line doesn't say) and none are given, so a run's fights end at the pair unless it takes all three itself on the way. Its own
arrivals had the whole run to take them, so read the Heart column across those two with that in mind.

Up to --limit lines a group: first the ones held out of training (`human_bosses.held_out`), then the rest by a hash of the play id, so every fighter
gets the same states. Fighters may have practised on winners' boss fights from the train file (`bosses-silent3`), but never on the placebos': a group's
row "never practised" is its held-out lines alone, the check on that (shown when the group has both kinds). A network trained with
`train_runs.py --late-starts` has played on from this very train file, placebos too, and by the same hash, so the --limit lines here are the ones its pool
held: for it only the "never practised" rows, or the eval file alone, are states it hasn't seen.

The dump is untrusted data: this only parses it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import collections
import glob
import gzip
import hashlib
import json
import math
import time

import numpy as np
import sts
import torch

from core.fight_model import pick_device
from core.lookup import Lookup, trained_events
from core.split import SplitPolicy
from core.sts_env import RunEnv
from human.human_bosses import held_out
from human.human_runs import CHARACTERS
from search.search_eval import play
from training.train_runs import SCENARIO_EVAL_SEED, load, load_fight_net

GROUPS = {"winners": ("heart", "won"), "finishers": ("died3", "died4"), "act2-deaths": ("died2",), "heart": ("heart",), "won": ("won",),
          "act1-deaths": ("died1",), "act3-deaths": ("died3",), "act4-deaths": ("died4",)}
NAMES = {"winners": "winners", "finishers": "other finishers", "act2-deaths": "Act 2 deaths", "heart": "beat the Heart", "won": "the pair, no Heart",
         "act1-deaths": "Act 1 deaths", "act3-deaths": "Act 3 deaths", "act4-deaths": "final act deaths"}
BOSS_FLOOR = {"boss1": 16, "boss2": 33, "boss3": 50}
RATES = (("beat", "beat boss"), ("next", "next act"), ("act3", "Act 3"), ("pair", "the pair"), ("heart", "the Heart"))
READ_ON = ("next", "pair")  # the plan's measures, compared across groups at the end of each kind


def label_of(s):
    o = s["outcome"]
    return o["label"] + str(o.get("died_act", ""))


def read_lines(paths, kinds, limit):
    """({(kind, label): [(sort key, line)]}, how many lines of each there were): up to `limit` lines of each kept, held out of training first, then by a
    hash of the play id."""
    cells, counts = collections.defaultdict(list), collections.Counter()
    for path in paths:
        with (gzip.open if path.endswith(".gz") else open)(os.path.expanduser(path), "rt") as f:
            for line in f:
                s = json.loads(line)
                if "outcome" not in s:
                    raise SystemExit(f"{path}: its lines don't say how the runs ended: human_bosses.py --all-outcomes makes them")
                kind = s.get("room", "boss") + str(s["act"])
                if kind in kinds:
                    counts[kind, label_of(s)] += 1
                    cell = cells[kind, label_of(s)]
                    cell.append(((not held_out(s["play_id"]), hashlib.md5(str(s["play_id"]).encode()).hexdigest()), line.strip()))
                    if len(cell) > 4 * limit:
                        cell.sort()
                        del cell[limit:]
    return cells, counts


def choose(cells, counts, kind, group, limit):
    """The group's lines at this boss, and how many it has in all."""
    found = [x for label in GROUPS[group] for x in cells.get((kind, label), [])]
    return [line for _, line in sorted(found)[:limit]], sum(counts[kind, label] for label in GROUPS[group])


def adjust(line, strip_relics, full_hp):
    """The state with its character's starter relic only, and on full HP, as asked."""
    s = json.loads(line)
    if strip_relics:
        s["relics"] = [CHARACTERS[s["character"]][1]]
    if full_hp:
        s["hp"] = s["max_hp"]
    return json.dumps(s)


def make_policy(args, device, amp):
    """(the network to play with, its lookup or None, what plays what)."""
    net, ck = load(args.checkpoint, device)
    net.load_state_dict(ck["net"])
    net.eval()
    own = ck["args"].get("fight_net")
    fight_path = args.fighter or own
    if not fight_path:
        return net, None, f"{args.checkpoint} (update {ck.get('update')}) plays everything"
    lookup = None
    if ck["args"].get("lookup"):
        lookup = Lookup(load_fight_net(own, device), device, amp, draws=ck["args"].get("lookup_draws", 4),
                        every=ck["args"].get("lookup_every", False), events=trained_events(ck["net"]))
    return SplitPolicy(load_fight_net(fight_path, device), net), lookup, (f"fights: {fight_path}; choices: {args.checkpoint} (update {ck.get('update')}), "
                                                                         f"{'with its lookup' if lookup else 'no lookup'}")


def play_states(policy, lookup, device, amp, lines, args, label):
    """Plays each line once on, greedily; the run summaries, in the lines' order."""
    n = len(lines)
    b = min(args.batch, n)
    env = RunEnv(b, seed=args.seed, ascension=20, practice=1.0, max_steps=8000, character=args.character)
    if args.free_keys:
        env.envs.set_free_keys(True)
    skipped = env.envs.set_scenarios(lines, 1.0, True, True)
    assert skipped == 0, f"{skipped} lines not of this character"
    out, _ = play(policy, device, amp, env, lambda g, e: e * b + g < n, None, 0.0, label, lookup=lookup)
    out.sort(key=lambda e: e["scenario"])
    assert [e["scenario"] for e in out] == list(range(n))
    return out


def measures(eps, floor, act):
    """Each rate as (share, standard error), the floor as (mean, standard error), and the keys, from run summaries that reached `floor`, `act`'s boss."""
    def rate(flags):
        p = float(np.mean(flags)) if len(flags) else float("nan")
        return p, math.sqrt(p * (1 - p) / max(len(flags), 1))

    floors = [e["floor"] for e in eps]
    return {
        "n": len(eps),
        "beat": rate([e["won"] or e["floor"] > floor for e in eps]),
        "next": rate([e["won"] or e["act"] > act for e in eps]),
        "act3": rate([e["won"] or e["act"] >= 3 for e in eps]),
        "pair": rate([e["won"] or e["act"] >= 4 for e in eps]),
        "heart": rate([e["won"] and e["act"] >= 4 for e in eps]),
        "floor": (float(np.mean(floors)), float(np.std(floors) / math.sqrt(len(floors)))) if eps else (float("nan"), 0.0),
        "keys": float(np.mean([e.get("keys", 0) for e in eps])) if eps else float("nan"),
        "three_keys": float(np.mean([e.get("keys", 0) == 3 for e in eps])) if eps else float("nan"),
        "timed_out": sum(bool(e.get("timed_out")) for e in eps),
    }


def row(name, m, state=""):
    rates = "  ".join(f"{100 * m[k][0]:5.1f} ±{100 * m[k][1]:3.1f}" for k, _ in RATES)
    late = f", {m['timed_out']} timed out" if m["timed_out"] else ""
    return (f"  {name:28s} {m['n']:5d}  {rates}  {m['floor'][0]:4.1f} ±{m['floor'][1]:3.1f}  {m['keys']:4.2f} {100 * m['three_keys']:4.1f}%{state}{late}")


def state_of(states):
    """What the states brought, for a row."""
    if not states:
        return ""
    mean = lambda f: float(np.mean([f(s) for s in states]))  # noqa: E731
    return (f"  | brought HP {100 * mean(lambda s: s['hp'] / max(s['max_hp'], 1)):.0f}%, deck {mean(lambda s: len(s['deck'])):.1f}, "
            f"relics {mean(lambda s: len(s['relics'])):.1f}, potions {mean(lambda s: len(s['potions'])):.1f}, gold {mean(lambda s: s.get('gold') or 0):.0f}")


def own_runs(prefixes):
    """The run summaries of search_eval.py --dumps (<prefix>.<process>.jsonl), the prefixes comma-separated."""
    runs = []
    for prefix in prefixes.split(","):
        paths = sorted(glob.glob(f"{os.path.expanduser(prefix)}.*.jsonl"))
        if not paths:
            raise SystemExit(f"no {prefix}.*.jsonl")
        for path in paths:
            with open(path) as f:
                runs += [json.loads(line) for line in f]
    return runs


def own_arrivals(runs, floor):
    """The runs that reached this floor: on a boss floor, at that boss."""
    return [e for e in runs if e["floor"] >= floor]


def reading(results, own, group_names, who="winners"):
    """Lines comparing winners (`who`: which of them are in `results`) with its own arrivals and the placebos, on the plan's measures."""
    out = []
    winners = results.get("winners")
    if winners is None:
        return out
    for k, name in RATES:
        if k not in READ_ON:
            continue
        p, se = winners[k]
        parts = [f"{who} {100 * p:.1f}"]
        if own is not None and own[k][0] > 0:
            ratio, q = p / own[k][0], own[k][0]
            parts.append(f"{ratio:.2f} ±{ratio * math.hypot(se / max(p, 1e-9), own[k][1] / q):.2f} x its own {100 * q:.1f}")
        for g in group_names:
            if g != "winners" and g in results:
                q, sq = results[g][k]
                d = p - q
                parts.append(f"{100 * d:+.1f} on {NAMES[g].lower()} ({d / max(math.hypot(se, sq), 1e-9):+.1f} SE)")
        out.append(f"  {name}: " + "; ".join(parts))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="makes the choices between fights (a run network or a deck network)")
    p.add_argument("lines", nargs="+", help="human_bosses.py --all-outcomes files")
    p.add_argument("--fighter", default=None, help="a fight network's checkpoint to play the fights, on any checkpoint")
    p.add_argument("--own", default=None, help="search_eval.py --dump prefixes, comma-separated: this fighter's own runs, for the rates from its own arrivals")
    p.add_argument("--kinds", default="boss1,boss2", help="which bosses' states: boss1, boss2 (boss3 has only those who fought both)")
    p.add_argument("--groups", default="winners,finishers,act2-deaths", help=f"of {','.join(GROUPS)}")
    p.add_argument("--limit", type=int, default=1000, help="states at most a group, at each boss")
    p.add_argument("--strip-relics", action="store_true", help="each state with its starter relic only")
    p.add_argument("--full-hp", action="store_true", help="each state on full HP")
    p.add_argument("--free-keys", action="store_true", help="each state holds the three keys: a run that beats both Act 3 bosses goes on to the Heart")
    p.add_argument("--character", default="silent", choices=["ironclad", "silent", "defect", "watcher"])
    p.add_argument("--batch", type=int, default=256, help="runs at once")
    p.add_argument("--seed", type=int, default=SCENARIO_EVAL_SEED, help="state j of a group plays on seed --seed + j")
    p.add_argument("--gpu-gb", type=float, default=3.5, help="GPU memory cap, GB (past it the process fails instead of spilling; 0 = none)")
    p.add_argument("--device", default="auto")
    p.add_argument("--dump", default=None, help="write every run's summary, with its group and her result, to this file")
    args = p.parse_args()
    kinds, group_names = args.kinds.split(","), args.groups.split(",")
    for g in group_names:
        if g not in GROUPS:
            raise SystemExit(f"no group {g}: {', '.join(GROUPS)}")
    device = pick_device(args.device)
    if device.type == "cuda" and args.gpu_gb:
        index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_per_process_memory_fraction(min(args.gpu_gb * 2 ** 30 / torch.cuda.get_device_properties(index).total_memory, 1.0), index)

    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    policy, lookup, who = make_policy(args, device, amp)
    print(who, flush=True)
    print(("each state holds the three keys (--free-keys)" if args.free_keys else "no state holds a key and none are given: its fights end at the pair unless it takes all three on the way")
          + ("; each state has its starter relic only (--strip-relics)" if args.strip_relics else "") + ("; each state on full HP (--full-hp)" if args.full_hp else ""), flush=True)
    cells, counts = read_lines(args.lines, set(kinds), args.limit)
    runs = own_runs(args.own) if args.own else None
    if args.own:
        print(f"{args.own}: {len(runs)} of its own runs, keys at the end {np.mean([e.get('keys', 0) for e in runs]) if runs else 0:.2f} a run", flush=True)
    dump = open(args.dump, "w") if args.dump else None
    header = "  ".join(f"{name:>10s}" for _, name in RATES)
    for kind in kinds:
        act, floor = int(kind[-1]), BOSS_FLOOR[kind]
        results, rows, never_practised = {}, [], {}
        for group in group_names:
            lines, found = choose(cells, counts, kind, group, args.limit)
            problems = collections.Counter(filter(None, map(sts.scenario_problem, lines)))
            if problems:
                lines = [line for line in lines if not sts.scenario_problem(line)]
                print(f"  {kind} {group}: left out, the engine can't play them: {dict(problems)}", flush=True)
            if not lines:
                rows.append(f"  {NAMES[group]:28s} no states ({found} read)")
                continue
            t0 = time.time()
            played = [adjust(x, args.strip_relics, args.full_hp) for x in lines]
            eps = play_states(policy, lookup, device, amp, played, args, f"{kind} {group}")
            states = [json.loads(x) for x in lines]
            print(f"{kind} {group}: {len(lines)} of {found} states played, {time.time() - t0:.0f} s", flush=True)
            results[group] = measures(eps, floor, act)
            rows.append(row(NAMES[group], results[group], state_of([json.loads(x) for x in played])))
            # The rows inside a group: its results (winners') and the states no training saw.
            for label in GROUPS[group] if len(GROUPS[group]) > 1 else ():
                part = [e for e, s in zip(eps, states) if label_of(s) == label]
                rows.append(row("  " + {"heart": "beat the Heart", "won": "the pair, no Heart", "died3": "died in Act 3", "died4": "died in the final act"}[label], measures(part, floor, act)))
            unseen = [e for e, s in zip(eps, states) if held_out(s["play_id"])]
            if 0 < len(unseen) < len(eps):
                never_practised[group] = measures(unseen, floor, act)
                rows.append(row("  never practised", never_practised[group]))
            if dump:
                for e, s in zip(eps, states):
                    dump.write(json.dumps({"kind": kind, "group": group, "stripped": args.strip_relics, "full_hp": args.full_hp, "play_id": s["play_id"], "outcome": s["outcome"], "run": e}) + "\n")
                dump.flush()
        print(f"\n{kind} (floor {floor}), % ± standard error{', her relics stripped' if args.strip_relics else ''}{', on full HP' if args.full_hp else ''}:")
        print(f"  {'':28s} {'runs':>5s}  {header}  {'floor':>10s}  keys  3 keys")
        for r in rows:
            print(r)
        own = None
        if runs is not None:
            arrivals = own_arrivals(runs, floor)
            own = measures(arrivals, floor, act)
            print(row(f"own arrivals ({len(arrivals)} of {len(runs)})", own))
        for line in reading(results, own, group_names):
            print(line)
        if "winners" in never_practised:
            for line in reading({**results, "winners": never_practised["winners"]}, own, group_names, "winners never practised"):
                print(line)
        print(flush=True)
    if dump:
        dump.close()


if __name__ == "__main__":
    main()
