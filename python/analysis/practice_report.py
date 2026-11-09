"""Where a run network's practice fights are lost, and how it plays them.

    .venv/bin/python python/analysis/practice_report.py <checkpoint> [--fights 128] [--retries 8] [--out report.json]

Plays fixed practice fights (the checkpoint run's --practice-decks: --fights of each deck kind in
each act, A0-20 drawn per fight) greedily, keeping every combat move's state (`Envs(detail=True)`),
and reports:
- the win rate and HP lost by act and room (hallway, elite, boss), deck kind and engine,
  ascension, and the encounters lost most;
- how fights are lost: the turn of death, the damage taken a turn, the attacks left unblocked,
  potions still held at death, fights that ran into the turn limit;
- how it plays, by engine: what it plays (attacks, skills, powers), the orbs it holds, turns ended
  with energy and a playable card left, block against the attacks coming, potions, and the cards
  it plays most and least when they are in hand;
- with --retries, each lost fight (up to --replay-max) replayed with moves sampled from the
  network: a loss that some replay wins was winnable with this deck.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import contextlib
import json
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from core.fight_model import masked
from core.sts_env import RunEnv
from training.train_runs import DECK_ACTS, DECK_KINDS, deck_kind, load

SEED = 800_000
ASC_BANDS = ((0, 5), (6, 10), (11, 15), (16, 20))


def policy(net, device, amp, obs, sample):
    """Moves for a batch of observations (greedy, or sampled), and each row's policy entropy."""
    ints = torch.from_numpy(obs["ints"]).to(device).long()
    floats = torch.from_numpy(obs["floats"]).to(device)
    mask = torch.from_numpy(obs["mask"]).to(device)
    with torch.no_grad(), amp():
        logits, _ = net(ints, floats, torch.ones(len(ints), dtype=torch.bool, device=device))
    logp = F.log_softmax(masked(logits.float(), mask), -1)
    ent = -(logp.exp() * logp).nan_to_num(0.0).sum(-1)
    acts = torch.distributions.Categorical(logits=logp).sample() if sample else logp.argmax(-1)
    return acts.cpu().numpy(), ent.cpu().numpy()


def play_fixed(net, device, amp, character, decks, per, asc, batch=128):
    """`per` fixed practice fights of each deck kind in each act, greedy: their episodes (with
    each move's state) and the policy entropy of every decision."""
    eps, ents = [], []
    for i, kind in enumerate(DECK_KINDS if decks == "mix" else [decks]):
        for act in DECK_ACTS:
            n = min(batch, per)
            env = RunEnv(n, seed=SEED + 16 * i + act, practice=1.0, max_steps=2000, character=character, practice_decks=kind,
                         practice_act=act, ascension_range=asc, record=True, detail=True)
            done, got = np.zeros(n, dtype=int), []
            per_game = -(-per // n)
            while (done < per_game).any():
                obs = env.observe()
                acts, ent = policy(net, device, amp, obs, False)
                ents.append(ent)
                for g in np.flatnonzero(env.step(acts)):
                    if done[g] < per_game:
                        ep = json.loads(env.last_episode(int(g)))
                        ep["kind"], ep["seed_kind"] = deck_kind(ep["practice"]), kind
                        got.append(ep)
                    done[g] += 1
            eps += got[:per]
            print(f"  {kind} act {act}: {sum(e['won'] for e in got[:per])} of {len(got[:per])} won", flush=True)
    return eps, np.concatenate(ents)


def replay(net, device, amp, character, losses, retries, asc):
    """Each lost fight played `retries` times more with sampled moves (the same fight: its seed,
    deck kind and act): the share of them that some replay won, and of all the replays."""
    envs = []
    for e in losses:
        for _ in range(retries):
            envs.append((e, RunEnv(1, seed=e["seed"], practice=1.0, max_steps=2000, character=character,
                                   practice_decks=e["seed_kind"], practice_act=e["act"], ascension_range=asc)))
    won = defaultdict(list)
    live = list(range(len(envs)))
    while live:
        obs = [envs[k][1].observe() for k in live]
        batch = {key: np.concatenate([o[key] for o in obs]) for key in ("ints", "floats", "mask")}
        acts, _ = policy(net, device, amp, batch, True)
        still = []
        for j, k in enumerate(live):
            status = envs[k][1].step([acts[j]])[0]
            if status:
                won[id(envs[k][0])].append(status == 1)
            else:
                still.append(k)
        live = still
    winnable = [any(won[id(e)]) for e in losses]
    return winnable, float(np.mean([w for v in won.values() for w in v])) if won else 0.0


def rate(eps):
    return (float(np.mean([e["won"] for e in eps])), len(eps)) if eps else (None, 0)


def hp_lost(eps):
    won = [e for e in eps if e["won"]]
    return float(np.mean([e["hp_start"] - e["hp"] for e in won])) if won else None


def by(eps, key):
    groups = defaultdict(list)
    for e in eps:
        groups[key(e)].append(e)
    return {g: {"won": rate(v)[0], "fights": len(v), "hp_lost": hp_lost(v)} for g, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def room(e):
    r = e.get("room") or "Monster"
    return {"Monster": "hallway", "Elite": "elite", "Boss": "boss"}.get(r, r.lower())


def band(a):
    return next(f"A{lo}-{hi}" for lo, hi in ASC_BANDS if lo <= a <= hi)


def turns_of(e):
    """The fight's moves with their state, grouped by turn: [(turn, [moves])]."""
    out = defaultdict(list)
    for m in e["log"]["steps"]:
        if "state" in m:
            out[m["state"]["turn"]].append(m)
    return sorted(out.items())


def play_stats(eps):
    """How the fights in `eps` were played (see the module's docstring)."""
    plays, kinds, powers_played, orbs, offered, played = Counter(), Counter(), [], Counter(), Counter(), Counter()
    turn_ends = wasted = 0
    unblocked, over, block_when_safe, attacked_turns, safe_turns = [], [], 0, 0, 0
    drunk, drunk_turn, potions_left_lost = Counter(), [], []
    focus_fire = multi_target = 0
    for e in eps:
        n_powers = 0
        for _, moves in turns_of(e):
            first = moves[0]["state"]
            in_hand = Counter(c[0] for c in first["hand"] if c[3])
            this_turn = Counter()
            for m in moves:
                s, mv = m["state"], m["state"]["move"]
                if mv.get("card"):
                    name, ctype = mv["card"][0], mv["card"][1]
                    plays[name] += 1
                    kinds[ctype] += 1
                    this_turn[name] += 1
                    n_powers += ctype == "Power"
                    if ctype == "Attack" and mv.get("target") and len(s["monsters"]) > 1:
                        multi_target += 1
                        lowest = min(s["monsters"], key=lambda x: x[1] + x[2])[0]
                        focus_fire += mv["target"] == lowest
                if mv.get("potion"):
                    drunk[mv["potion"]] += 1
                    drunk_turn.append(s["turn"])
                if mv.get("end"):
                    turn_ends += 1
                    wasted += s["energy"] > 0 and any(c[3] for c in s["hand"])
                    if s["incoming"] > 0:
                        attacked_turns += 1
                        unblocked.append(max(0, s["incoming"] - s["block"]) / max(s["incoming"], 1))
                        over.append(max(0, s["block"] - s["incoming"]))
                    else:
                        safe_turns += 1
                        block_when_safe += s["block"] > 0
                    for o in s["orbs"]:
                        orbs[o] += 1  # "Empty" for an empty slot
            for name, k in in_hand.items():
                offered[name] += k
                played[name] += min(k, this_turn[name])
        powers_played.append(n_powers)
        if not e["won"]:
            potions_left_lost.append(e.get("potions", 0))
    total = sum(kinds.values()) or 1
    empty = orbs.pop("Empty", 0)
    orb_total = sum(orbs.values()) or 1
    rates = {c: (played[c] / offered[c], offered[c]) for c in offered if offered[c] >= 30}
    return {
        "plays_per_fight": sum(kinds.values()) / max(len(eps), 1),
        "card_types": {k: v / total for k, v in kinds.most_common()},
        "powers_per_fight": float(np.mean(powers_played)) if powers_played else 0.0,
        "orbs_held": {k: v / orb_total for k, v in orbs.most_common()},
        "orbs_per_turn_end": sum(orbs.values()) / max(turn_ends, 1),
        "empty_orb_slots_per_turn_end": empty / max(turn_ends, 1),
        "turns_wasting_energy": wasted / max(turn_ends, 1),
        "attacked_turns_unblocked": float(np.mean(unblocked)) if unblocked else None,
        "attacked_turns_fully_blocked": float(np.mean([u == 0 for u in unblocked])) if unblocked else None,
        "overblock_when_attacked": float(np.mean(over)) if over else None,
        "safe_turns_with_block": block_when_safe / max(safe_turns, 1),
        "potions_drunk_per_fight": sum(drunk.values()) / max(len(eps), 1),
        "potion_turn_median": float(np.median(drunk_turn)) if drunk_turn else None,
        "potions_drunk": dict(drunk.most_common(8)),
        "potions_held_at_death": float(np.mean(potions_left_lost)) if potions_left_lost else None,
        "attacks_on_lowest_hp_target": focus_fire / max(multi_target, 1),
        "most_played": [(c, round(r, 2), n) for c, (r, n) in sorted(rates.items(), key=lambda kv: -kv[1][0])[:12]],
        "least_played": [(c, round(r, 2), n) for c, (r, n) in sorted(rates.items(), key=lambda kv: kv[1][0])[:12]],
    }


def loss_anatomy(eps):
    """How won and lost fights go: turns, damage taken a turn, the turn of death."""
    out = {}
    for name, sub in (("won", [e for e in eps if e["won"]]), ("lost", [e for e in eps if not e["won"]])):
        turns, taken = [], []
        for e in sub:
            ts = turns_of(e)
            turns.append(len(ts))
            hp = [moves[0]["state"]["hp"] for _, moves in ts] + [e["hp"]]
            taken += [max(0, a - b) for a, b in zip(hp, hp[1:])]
        out[name] = {"fights": len(sub), "turns_median": float(np.median(turns)) if turns else None,
                     "turns_p90": float(np.percentile(turns, 90)) if turns else None,
                     "damage_taken_per_turn": float(np.mean(taken)) if taken else None,
                     "timed_out": sum(e["timed_out"] for e in sub)}
    lost = [e for e in eps if not e["won"]]
    out["lost_by_turn"] = dict(sorted(Counter(min(len(turns_of(e)), 15) for e in lost).items()))
    out["lost_start_hp_share"] = float(np.mean([e["hp_start"] / max(e["max_hp"], 1) for e in lost])) if lost else None
    out["won_start_hp_share"] = float(np.mean([e["hp_start"] / max(e["max_hp"], 1) for e in eps if e["won"]])) if eps else None
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--fights", type=int, default=128, help="fixed fights of each deck kind in each act")
    p.add_argument("--retries", type=int, default=8, help="sampled replays of each lost fight (0: none)")
    p.add_argument("--replay-max", type=int, default=400, help="lost fights to replay at most")
    p.add_argument("--ascension-range", type=int, nargs=2, default=(0, 20))
    p.add_argument("--out", default=None, help="JSON report (default: next to the checkpoint)")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else contextlib.nullcontext
    net, ck = load(a.checkpoint, device)
    net.load_state_dict(ck["net"])
    net.eval()
    args = ck["args"]
    character, decks = args.get("character", "ironclad"), args.get("practice_decks") or "mix"
    asc = tuple(a.ascension_range)
    print(f"{a.checkpoint} (update {ck.get('update')}): {character}, {decks} decks, {a.fights} fights of each kind in each act, A{asc[0]}-{asc[1]}")
    eps, ents = play_fixed(net, device, amp, character, decks, a.fights, asc)

    report = {"checkpoint": a.checkpoint, "update": ck.get("update"), "fights": len(eps), "won": rate(eps)[0], "hp_lost": hp_lost(eps),
              "policy_entropy": {"mean": float(ents.mean()), "share_under_0.1": float((ents < 0.1).mean()), "share_over_1": float((ents > 1).mean())},
              "by_act_room": by(eps, lambda e: f"act {e['act']} {room(e)}"),
              "by_kind": by(eps, lambda e: e["kind"]),
              "by_engine": by([e for e in eps if e["kind"] == "engine"], lambda e: e["practice"]),
              "by_ascension": by(eps, lambda e: band(e["log"]["ascension"])),
              "anatomy": loss_anatomy(eps)}
    enc = by(eps, lambda e: (e["act"], e["encounter"]))
    report["worst_encounters"] = sorted(([f"act {k[0]} {k[1]}", v["won"], v["fights"]] for k, v in enc.items() if v["fights"] >= 8),
                                        key=lambda x: x[1])[:15]
    report["play"] = {"all": play_stats(eps), "won": play_stats([e for e in eps if e["won"]]), "lost": play_stats([e for e in eps if not e["won"]])}
    report["play_by_engine"] = {g: play_stats([e for e in eps if e["practice"] == g]) for g in sorted({e["practice"] for e in eps})}
    if a.retries:
        losses = [e for e in eps if not e["won"]][:a.replay_max]
        print(f"replaying {len(losses)} lost fights {a.retries} times each with sampled moves", flush=True)
        winnable, replay_won = replay(net, device, amp, character, losses, a.retries, asc)
        for e, w in zip(losses, winnable):
            e["winnable"] = w
        report["replays"] = {"lost_fights": len(losses), "winnable": float(np.mean(winnable)) if winnable else None, "replays_won": replay_won,
                             "winnable_by_act": {act: float(np.mean([e["winnable"] for e in losses if e["act"] == act]))
                                                 for act in DECK_ACTS if any(e["act"] == act for e in losses)},
                             "winnable_by_kind": {k: float(np.mean([e["winnable"] for e in losses if e["kind"] == k]))
                                                  for k in DECK_KINDS if any(e["kind"] == k for e in losses)}}
    out = a.out or os.path.join(os.path.dirname(a.checkpoint), f"practice_report_{ck.get('update')}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("play_by_engine",)}, indent=1))
    print("play by engine:")
    for g, st in report["play_by_engine"].items():
        print(f"  {g}: {json.dumps({k: st[k] for k in ('plays_per_fight', 'card_types', 'powers_per_fight', 'orbs_held', 'turns_wasting_energy', 'attacked_turns_unblocked', 'potions_drunk_per_fight')})}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
