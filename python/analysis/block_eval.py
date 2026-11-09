"""How well a network plays block decks: greedy play of fixed block practice fights, with win rate,
HP left and the block-specific habits.

    python python/analysis/block_eval.py <checkpoint> [--fights 512] [--ascension 0]

Habits counted (in fights of 4+ turns where noted):
- barricade: share of fights of 4+ turns in which Barricade was playable that it was played, and
  how many turns after it first was; barricade_early: the same for fights of 5+ turns in which it
  was playable by turn 2 (where skipping it is clearly wrong);
- slam_zero: Body Slams played with no block; slam_order: Body Slams followed the same turn by a
  block card that was already in hand and affordable (blocking first deals more);
- imp_waste: Impervious played with no attack coming and no Barricade (its block falls off);
- held_barricade: turns ended under Barricade with energy left and a block card it could play;
- held_hurt: turns ended taking unblocked damage with energy left and a block card it could play;
- entrench_low: Entrench played with under 10 block.
train_runs.py runs it with --practice-archetype block (block.csv).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import collections
import json

import numpy as np
import torch

from core.fight_model import masked, pick_device
from core.sts_env import LAYOUT, RunEnv

NAMES = LAYOUT["card_names"]
BLOCK = {"Defend", "Shrug It Off", "Impervious", "Iron Wave", "Flame Barrier", "Ghostly Armor", "Power Through",
         "True Grit", "Second Wind", "Sentinel"}
HF = 10
END_TURN = 105
BLOCK_EVAL_SEED = 7_000_000
FIELDS = ["update", "steps", "fights", "win_rate", "hp_left", "barricade", "barricade_early", "barricade_wait", "slam_zero", "slam_order", "slam_block",
          "imp_waste", "held_barricade", "held_hurt", "entrench_low", "turns"]


def name(i):
    return NAMES[i - 1] if i else None


def block_eval(net, device, fights=512, amp=None, ascension=0, batch=128, archetype="block", seed=BLOCK_EVAL_SEED):
    net.eval()
    env = RunEnv(batch, seed=seed, ascension=ascension, practice=1.0, practice_archetype=archetype, max_steps=2000)
    per_game = -(-fights // batch)
    done = np.zeros(batch, dtype=int)
    c = collections.Counter()
    slam_block, waits, turns = [], [], []
    bar_seen = [None] * batch; bar_played = [None] * batch; slam_turn = [None] * batch; last_turn = [0] * batch
    slam_held = [collections.Counter() for _ in range(batch)]  # block cards in hand and affordable at the last Body Slam
    out = []

    def close(g):
        if done[g] < per_game:
            turns.append(last_turn[g])
            if bar_seen[g] is not None and last_turn[g] >= 4:
                c["bar_fights"] += 1
                if bar_played[g] is not None:
                    c["bar_played"] += 1
                    waits.append(bar_played[g] - bar_seen[g])
            if bar_seen[g] is not None and bar_seen[g] <= 2 and last_turn[g] >= 5:
                c["bar_early_fights"] += 1
                c["bar_early_played"] += bar_played[g] is not None
        bar_seen[g] = bar_played[g] = slam_turn[g] = None
        slam_held[g].clear()
        last_turn[g] = 0

    amp = amp or (lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"))
    while (done < per_game).any():
        obs = env.observe()
        ints = torch.from_numpy(obs["ints"]).to(device).long()
        floats = torch.from_numpy(obs["floats"]).to(device)
        mask = torch.from_numpy(obs["mask"]).to(device)
        prac = torch.ones(batch, dtype=torch.bool, device=device)
        with torch.no_grad(), amp():
            logits, _ = net(ints, floats, prac)
        acts = masked(logits.float(), mask).argmax(-1).cpu().numpy()
        for g in np.flatnonzero(done < per_game):
            s = obs["f_scalars"][g]
            block, energy, turn = s[2], s[3], int(s[5])
            if turn < 1:
                continue
            last_turn[g] = turn
            hand = obs["i_hand_card"][g]
            f = obs["f_hand"][g].reshape(10, HF)
            cards = [(name(hand[h]), f[h]) for h in range(10) if hand[h]]
            playable = [(n, r) for n, r in cards if r[5] > 0]
            mon = obs["f_monster"][g].reshape(-1, 8)
            incoming = float((mon[:, 3] * mon[:, 4]).sum())
            if bar_seen[g] is None and any(n == "Barricade" for n, _ in playable):
                bar_seen[g] = turn
            a = int(acts[g])
            if a < 70 and hand[a // 7]:
                n, r = name(hand[a // 7]), f[a // 7]
                if n == "Barricade" and bar_played[g] is None:
                    bar_played[g] = turn
                elif n == "Body Slam":
                    c["slams"] += 1
                    c["slam_zero"] += block == 0
                    slam_block.append(block)
                    slam_turn[g] = turn
                    after = energy - r[0]
                    slam_held[g] = collections.Counter(m for m, rr in playable if m in BLOCK and rr[0] <= after)
                elif n == "Impervious":
                    c["imps"] += 1
                    c["imp_waste"] += incoming == 0 and bar_played[g] is None
                elif n == "Entrench":
                    c["entrenches"] += 1
                    c["entrench_low"] += block < 10
                if n in BLOCK and slam_turn[g] == turn and slam_held[g][n] > 0:
                    # A block card that was in hand (and affordable) when Body Slam went first.
                    slam_held[g][n] -= 1
                    c["slam_order"] += 1
            elif a == END_TURN:
                affordable = any(n in BLOCK and r[0] <= energy for n, r in playable)
                spare = energy > 0 and affordable
                if bar_played[g] is not None:
                    c["ends_barricade"] += 1
                    c["held_barricade"] += spare
                c["ends"] += 1
                c["held_hurt"] += spare and incoming > block
        status = env.step(acts)
        for g in np.flatnonzero(status):
            if done[g] < per_game:
                out.append(json.loads(env.last_episode(int(g))))
            close(g)
            done[g] += 1
    net.train()
    out = out[:fights]
    ratio = lambda a, b: c[a] / c[b] if c[b] else ""
    return {
        "fights": len(out),
        "win_rate": float(np.mean([e["won"] for e in out])),
        "hp_left": float(np.mean([0.5 + 0.5 * e["hp"] / max(e["max_hp"], 1) if e["won"] else 0.0 for e in out])),
        "barricade": ratio("bar_played", "bar_fights"),
        "barricade_early": ratio("bar_early_played", "bar_early_fights"),
        "barricade_wait": float(np.mean(waits)) if waits else "",
        "slam_zero": ratio("slam_zero", "slams"),
        "slam_order": ratio("slam_order", "slams"),
        "slam_block": float(np.mean(slam_block)) if slam_block else "",
        "imp_waste": ratio("imp_waste", "imps"),
        "held_barricade": ratio("held_barricade", "ends_barricade"),
        "held_hurt": ratio("held_hurt", "ends"),
        "entrench_low": ratio("entrench_low", "entrenches"),
        "turns": float(np.mean(turns)) if turns else "",
    }


def describe(r):
    pct = lambda k: f"{r[k]:.0%}" if r[k] != "" else "-"
    return (f"won {r['win_rate']:.1%}, hp_left {r['hp_left']:.3f}; Barricade played {pct('barricade')}, when playable by turn 2 of a 5+ turn fight {pct('barricade_early')} (wait {r['barricade_wait'] if r['barricade_wait'] == '' else round(r['barricade_wait'], 2)}); "
            f"Body Slam at 0 block {pct('slam_zero')}, before blocking {pct('slam_order')}, mean block {r['slam_block'] if r['slam_block'] == '' else round(r['slam_block'], 1)}; "
            f"Impervious wasted {pct('imp_waste')}; held block under Barricade {pct('held_barricade')}, while hurt {pct('held_hurt')}; "
            f"Entrench under 10 block {pct('entrench_low')}; {r['turns'] if r['turns'] == '' else round(r['turns'], 1)} turns")


def main():
    from training.train_runs import load
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--fights", type=int, default=512)
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("--seed", type=int, default=BLOCK_EVAL_SEED, help="fight seeds (another value gives fights no eval has seen)")
    p.add_argument("--archetype", default="block", help="practice archetype of the fights (block, exhaust, strength); "
                   "the habit counts are block's")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = pick_device(args.device)
    for path in args.checkpoints:
        net, ck = load(path, device)
        net.load_state_dict(ck["net"])
        print(f"{path} (update {ck['update']}) A{args.ascension}: {describe(block_eval(net, device, args.fights, ascension=args.ascension, seed=args.seed, archetype=args.archetype))}", flush=True)


if __name__ == "__main__":
    main()
