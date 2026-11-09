"""Why a fight network loses (jack, 2026-10-30: "grab a sample of fights that it loses. analyse and tell me why lose").

    .venv/bin/python python/analysis/fight_losses.py <checkpoint> --scenarios ~/sts-data/bosses-ironclad/eval.jsonl.gz \\
        [--limit N] [--retries 8] [--examples 3] [--out losses.json]

Plays held-out human fights (human_bosses.py's scenarios: a winner's deck, relics, HP and potions, A20) greedily,
keeping every move's state (`Envs(detail=True)`), and reports by kind of fight (Act 1-3 bosses, the Heart, Act 1-3
elites) and by encounter:
- the win rate, and how the lost fights went against the won ones: turns, damage taken a turn, the human's HP share
  and deck at the start (size, Strikes and Defends, curses), how much of the monsters' HP was gone at death, the
  blow that killed (the attack coming against block and HP on the last turn), potions never drunk, turns ended with
  energy and a playable card left, attacks left unblocked (practice_report.py's measures);
- each lost fight replayed --retries times with sampled moves and other draws: a loss that some replay wins was
  winnable with that deck, one that none wins was likely the deck (or play far off);
- --examples lost fights of each kind turn by turn (HP, block, energy, the attack coming, the hand, the moves), in
  a text file beside the report.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import contextlib
import json
import random
from collections import Counter, defaultdict

import numpy as np
import torch

from analysis.practice_report import loss_anatomy, play_stats, policy, turns_of
from core.sts_env import RunEnv
from training.train_runs import load, read_scenarios

SEED = 1_200_000
BASICS = ("Strike", "Defend")


def kind(f):
    room = f.get("room", "boss")
    return "the Heart" if f["act"] == 4 and room == "boss" else f"Act {f['act']} {room}"


def play(net, device, amp, lines, character, seed, batch, sample=False, record=True):
    """Each scenario line played once (greedy, or sampled): its episode, with each move's state if `record`."""
    out = []
    for at in range(0, len(lines), batch):
        chunk = lines[at:at + batch]
        n = len(chunk)
        env = RunEnv(n, seed=seed + at, practice=1.0, max_steps=4000, character=character, record=record, detail=record)
        if env.envs.set_scenarios(chunk, 1.0, True):
            raise ValueError("some scenarios couldn't be set up")
        res = [None] * n
        while any(r is None for r in res):
            acts, _ = policy(net, device, amp, env.observe(), sample)
            for g in np.flatnonzero(env.step(acts)):
                if res[g] is None:
                    res[g] = json.loads(env.last_episode(int(g)))
        out += res
        print(f"  {at + n:,} of {len(lines):,} fights", flush=True)
    return out


def deck_stats(f):
    """The human's deck: its size, Strikes and Defends, curses and upgraded cards (cards are [game id, upgrades])."""
    deck = [(c, 0) if isinstance(c, str) else (c[0], c[1]) for c in f["deck"]]
    return {"size": len(deck), "basics": sum(n.startswith(BASICS) for n, _ in deck), "curses": sum(n in CURSES for n, _ in deck),
            "upgraded": sum(u > 0 for _, u in deck)}


# The game's ids of the curses (as human runs write them).
CURSES = {"AscendersBane", "Clumsy", "CurseOfTheBell", "Decay", "Doubt", "Injury", "Necronomicurse", "Normality", "Pain",
          "Parasite", "Pride", "Regret", "Shame", "Writhe"}


def last_turn(e):
    """The state on the fight's last turn when it ended it (or its last move): HP, block, the attack coming, monsters."""
    ts = turns_of(e)
    if not ts:
        return None
    moves = ts[-1][1]
    end = next((m["state"] for m in reversed(moves) if m["state"]["move"].get("end")), moves[-1]["state"])
    return end


def monster_hp_gone(e):
    """The share of the monsters' HP (at the first move) gone at the fight's last move; a fight with monsters that
    split or come later (Slime Boss, Collector, Reptomancer) counts what was there at each moment."""
    ts = turns_of(e)
    if not ts:
        return None
    first = ts[0][1][0]["state"]["monsters"]
    last = ts[-1][1][-1]["state"]["monsters"]
    start = sum(m[1] for m in first)
    return max(0.0, 1 - sum(m[1] for m in last) / max(start, 1))


def undrunk(e, f):
    """The potions the human brought that the fight never drank."""
    left = Counter(f.get("potions", []))
    for _, moves in turns_of(e):
        for m in moves:
            p = m["state"]["move"].get("potion")
            if p and left[p] > 0:
                left[p] -= 1
    return [p for p, k in left.items() for _ in range(k)]


def compare(pairs):
    """Won against lost on the measures above, for a list of (episode, scenario)."""
    out = {}
    for name, sub in (("won", [(e, f) for e, f in pairs if e["won"]]), ("lost", [(e, f) for e, f in pairs if not e["won"]])):
        if not sub:
            continue
        decks = [deck_stats(f) for _, f in sub]
        ends = [last_turn(e) for e, _ in sub]
        ends = [s for s in ends if s]
        out[name] = {
            "fights": len(sub),
            "start_hp_share": float(np.mean([f["hp"] / f["max_hp"] for _, f in sub])),
            "deck_size": float(np.mean([d["size"] for d in decks])),
            "deck_basics": float(np.mean([d["basics"] for d in decks])),
            "deck_curses": float(np.mean([d["curses"] for d in decks])),
            "potions_brought": float(np.mean([len(f.get("potions", [])) for _, f in sub])),
            "potions_never_drunk": float(np.mean([len(undrunk(e, f)) for e, f in sub])),
            "monster_hp_gone": float(np.nanmean([monster_hp_gone(e) or np.nan for e, _ in sub])),
        }
        if name == "lost" and ends:
            # The killing blow: on the last turn, the attack coming beyond block against the HP left.
            lethal = [max(0, s["incoming"] - s["block"]) >= s["hp"] for s in ends]
            out[name]["last_turn"] = {"hp": float(np.mean([s["hp"] for s in ends])), "block": float(np.mean([s["block"] for s in ends])),
                                      "incoming": float(np.mean([s["incoming"] for s in ends])), "attack_was_lethal": float(np.mean(lethal)),
                                      "energy_left": float(np.mean([s["energy"] for s in ends]))}
            out[name]["timed_out"] = sum(e["timed_out"] for e, _ in sub)
            out[name]["never_drunk"] = Counter(p for e, f in sub for p in undrunk(e, f)).most_common(10)
    return out


def describe(e, f):
    """A lost fight turn by turn."""
    d = deck_stats(f)
    lines = [f"{kind(f)}: {f['bosses'][0]}  (floor {f.get('floor')}, HP {f['hp']}/{f['max_hp']}, deck {d['size']} cards with "
             f"{d['basics']} Strikes/Defends and {d['curses']} curses, potions {f.get('potions', [])}, relics {len(f.get('relics', []))})",
             f"  deck: {', '.join(c if isinstance(c, str) else c[0] + '+' * min(c[1], 1) + (str(c[1]) if c[1] > 1 else '') for c in f['deck'])}"]
    for turn, moves in turns_of(e):
        s = moves[0]["state"]
        mons = "; ".join(f"{m[0]} {m[1]} HP" + (f" +{m[2]} blk" if m[2] else "") + (f" attacks {m[3]}x{max(m[4], 1)}" if m[3] > 0 else "")
                         for m in s["monsters"])
        plays = []
        for m in moves:
            mv = m["state"]["move"]
            if mv.get("card"):
                plays.append(mv["card"][0] + (f">{mv['target']}" if mv.get("target") else ""))
            elif mv.get("potion"):
                plays.append(f"POTION {mv['potion']}" + (f">{mv['target']}" if mv.get("target") else ""))
            elif mv.get("chose"):
                plays.append(f"(chose {', '.join(mv['chose'])})")
            elif mv.get("end"):
                end = m["state"]
                plays.append(f"END [block {end['block']} vs {end['incoming']} coming, energy left {end['energy']}]")
        hand = ", ".join(c[0] for c in s["hand"])
        lines.append(f"  T{turn}: HP {s['hp']} blk {s['block']} E{s['energy']} | {mons}\n      hand: {hand}\n      -> {'; '.join(plays)}")
    lines.append(f"  lost{' (turn limit)' if e['timed_out'] else ''}; monster HP gone {100 * (monster_hp_gone(e) or 0):.0f}%, "
                 f"potions never drunk: {undrunk(e, f) or 'none'}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--scenarios", required=True)
    p.add_argument("--character", default="ironclad", choices=["ironclad", "silent", "defect", "watcher"])
    p.add_argument("--limit", type=int, default=0, help="a random sample of this many fights (0: all)")
    p.add_argument("--retries", type=int, default=8, help="sampled replays of each lost fight (0: none)")
    p.add_argument("--examples", type=int, default=3, help="lost fights of each kind written out turn by turn")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--out", default=None, help="JSON report (default: next to the checkpoint)")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else contextlib.nullcontext
    net, ck = load(a.checkpoint, device)
    net.load_state_dict(ck["net"])
    net.eval()

    fights = [json.loads(line) for line in read_scenarios(os.path.expanduser(a.scenarios))]
    if a.limit:
        fights = random.Random(0).sample(fights, min(a.limit, len(fights)))
    print(f"{a.checkpoint} (update {ck.get('update')}): {len(fights):,} held-out human fights", flush=True)
    eps = play(net, device, amp, [json.dumps(f) for f in fights], a.character, SEED, a.batch)
    pairs = list(zip(eps, fights))

    groups = defaultdict(list)
    for e, f in pairs:
        groups[kind(f)].append((e, f))
    order = ["Act 1 boss", "Act 2 boss", "Act 3 boss", "the Heart", "Act 1 elite", "Act 2 elite", "Act 3 elite"]
    report = {"checkpoint": a.checkpoint, "update": ck.get("update"), "fights": len(pairs),
              "won": float(np.mean([e["won"] for e in eps])), "all": compare(pairs),
              "by_kind": {k: compare(groups[k]) for k in order if k in groups}}
    enc = defaultdict(list)
    for e, f in pairs:
        enc[(kind(f), f["bosses"][0])].append(e["won"])
    report["by_encounter"] = sorted(([k[0], k[1], float(np.mean(v)), len(v)] for k, v in enc.items() if len(v) >= 10), key=lambda x: x[2])
    report["anatomy"] = {k: loss_anatomy([e for e, _ in groups[k]]) for k in order if k in groups}
    report["play"] = {"won": play_stats([e for e in eps if e["won"]]), "lost": play_stats([e for e in eps if not e["won"]])}

    lost = [(e, f) for e, f in pairs if not e["won"]]
    if a.retries and lost:
        print(f"replaying {len(lost):,} lost fights {a.retries} times each with sampled moves", flush=True)
        lines = [json.dumps(f) for _, f in lost for _ in range(a.retries)]
        rep = play(net, device, amp, lines, a.character, SEED + 7_919, a.batch, sample=True, record=False)
        wins = np.array([r["won"] for r in rep], dtype=float).reshape(len(lost), a.retries)
        winnable = wins.max(1) > 0
        report["replays"] = {"lost_fights": len(lost), "winnable": float(winnable.mean()), "replays_won": float(wins.mean()),
                             "by_kind": {k: {"winnable": float(np.mean([w for w, (_, f) in zip(winnable, lost) if kind(f) == k])),
                                             "replays_won": float(np.mean([r.mean() for r, (_, f) in zip(wins, lost) if kind(f) == k]))}
                                         for k in order if any(kind(f) == k for _, f in lost)}}
        for (e, _), w in zip(lost, winnable):
            e["winnable"] = bool(w)

    out = a.out or os.path.join(os.path.dirname(a.checkpoint), f"fight_losses_{ck.get('update')}.json")
    with open(out, "w") as fo:
        json.dump(report, fo, indent=1)
    with open(out.replace(".json", "_examples.txt"), "w") as fo:
        for k in order:
            sub = [(e, f) for e, f in lost if kind(f) == k]
            for e, f in random.Random(1).sample(sub, min(a.examples, len(sub))):
                fo.write(describe(e, f) + (f"\n  winnable in replays: {e.get('winnable')}" if "winnable" in e else "") + "\n\n")
    print(json.dumps({k: v for k, v in report.items() if k != "play"}, indent=1))
    print("play, won against lost:")
    for key in ("plays_per_fight", "card_types", "turns_wasting_energy", "attacked_turns_unblocked", "attacked_turns_fully_blocked",
                "overblock_when_attacked", "safe_turns_with_block", "potions_drunk_per_fight", "potion_turn_median", "potions_drunk",
                "attacks_on_lowest_hp_target"):
        print(f"  {key}: won {report['play']['won'].get(key)}  lost {report['play']['lost'].get(key)}")
    print(f"wrote {out} and {out.replace('.json', '_examples.txt')}")


if __name__ == "__main__":
    main()
