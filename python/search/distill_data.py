"""Searched fight positions for offline search distillation (jack, 2026-10-30: "search distillation (AlphaZero-style
expert iteration) ... lets try this. also optimise it"; distill_train.py trains on them).

    .venv/bin/python python/search/distill_data.py <checkpoint> --out ~/sts-data/distill/<name> --rollouts 16 \\
        [--envs 1024] [--steps 32] [--k 8] [--top 4] [--shrink 2] [--temp 0.03] [--roots 512] [--seed 1]

Plays practice fights as the checkpoint's own training did (train_runs.Collector with the checkpoint's arguments:
its practice decks, scenarios, elite share, ascension), with the fight look-ahead (fight_search.py) on every fight
move that has a choice (not a card selection): each candidate first move (the network's --top likeliest and ending
the turn) on --k redrawn copies, played on greedily for --turns turns and scored as practice fights are. Each searched
move gets a target over its candidates: the network's probabilities times exp(gain / --temp), each gain over the
network's own move shrunk towards 0 by --shrink standard errors (the copies' luck), so a move only gains where the
search is sure; and a clear win (a gain beyond --shrink standard errors) is played instead of the sampled move
(train_runs.py --search-play), so the positions and the returns follow the searched play.

Each rollout's searched positions go to a shard (shard_s<seed>_<i>.npz, compressed; several processes with other
--seed values can fill one folder): the arrays (ints as int16), the legal
moves, the move played, the PPO value targets (returns: GAE with the run's gamma and lambda, the network's values
bootstrapping), the candidates and the target, and the practice flags; stats.jsonl has each shard's counts and times.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import contextlib
import json
import time

import numpy as np
import torch

from core.fixed_shapes import FixedShapes
from training.train_runs import Collector, load, parse_args, rollout_buffers


def collector_args(ck, a):
    """The checkpoint's own training arguments with the look-ahead on and this script's sizes."""
    args = parse_args(["--name", "distill"])
    for k, v in ck["args"].items():
        if hasattr(args, k):
            setattr(args, k, v)
    args.envs, args.steps, args.seed = a.envs, a.steps, a.seed
    args.retry_losses = 0  # the fights as they come: replays would favour lost fights
    args.overlap = False
    args.search_weight, args.search_play, args.search_target = 1.0, True, "soft"
    args.search_share, args.search_k, args.search_top, args.search_turns, args.search_max_moves = 1.0, a.k, a.top, a.turns, a.max_moves
    args.search_shrink, args.search_temp, args.search_value, args.search_potion_value = a.shrink, a.temp, "fight", None
    args.search_roots, args.search_pass_rows = a.roots, a.pass_rows
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--k", type=int, default=8, help="redrawn copies per candidate")
    p.add_argument("--top", type=int, default=4, help="the network's likeliest moves tried (and ending the turn)")
    p.add_argument("--turns", type=int, default=1, help="turns each copy is played on (2: 88.1%% of the held-out human fights at 32 copies "
                                                         "against 85.0%% for 1 turn and 83.3%% greedy, ironclad-fights-3x128-elites update 2000)")
    p.add_argument("--max-moves", type=int, default=30, help="moves at most in a copy's playout")
    p.add_argument("--shrink", type=float, default=2.0, help="gains shrink towards 0 by this many standard errors")
    p.add_argument("--temp", type=float, default=0.03, help="a move's odds times e for each this much its score gains (value units)")
    p.add_argument("--roots", type=int, default=512, help="positions searched at once (RAM: about 40 copies of a run each)")
    p.add_argument("--pass-rows", type=int, default=8192, help="copies per network pass")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else contextlib.nullcontext
    net, ck = load(a.checkpoint, device)
    net.load_state_dict(ck["net"])
    net.eval()
    args = collector_args(ck, a)
    policy = FixedShapes(net, cuda_graphs=device.type == "cuda") if args.fixed_shapes else net
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"args_s{a.seed}.json"), "w") as f:
        json.dump({"checkpoint": a.checkpoint, "update": ck.get("update"), **vars(a)}, f, indent=1)
    # The checkpoint's reward scale (train_runs.py --scale-rewards): the search scores practice endings in its value's units.
    col = Collector(args, device, amp, a.seed, ck.get("reward_scale") if args.scale_rewards else None)
    out = rollout_buffers(args.steps, args.envs, device)
    # Shards named by --seed, so several processes (other seeds) can fill one folder.
    first = len([f for f in os.listdir(a.out) if f.startswith(f"shard_s{a.seed}_")])
    for i in range(first, first + a.rollouts):
        t0 = time.time()
        with torch.no_grad():
            ro = col.collect(i, policy, out)
        s = ro["searched"]
        rows = (out["search_slot"][..., 0] >= 0).reshape(-1)
        keep = torch.nonzero(rows).squeeze(1)
        B = args.steps * args.envs
        arrays = {"ints": out["ints"].reshape(B, -1)[keep].to(torch.int16), "floats": out["floats"].reshape(B, -1)[keep],
                  "mask": out["mask"].reshape(B, -1)[keep], "act": out["act"].reshape(B)[keep], "ret": out["ret"].reshape(B)[keep],
                  "val": out["val"].reshape(B)[keep], "slot": out["search_slot"].reshape(B, -1)[keep].to(torch.int16),
                  "p": out["search_p"].reshape(B, -1)[keep], "prac": out["prac"].reshape(B)[keep]}
        t1 = time.time()
        np.savez_compressed(os.path.join(a.out, f"shard_s{a.seed}_{i}.npz"), **{k: v.cpu().numpy() for k, v in arrays.items()})
        stat = {"seed": a.seed, "shard": i, "positions": B, "searched": s["n"], "kept": len(keep), "changed": s["changed"], "played": s["played"],
                "search_s": round(s["s"], 1), "rollout_s": round(ro["rollout_s"], 1), "save_s": round(time.time() - t1, 1),
                "practice_fights": len(ro["practice_now"]), "practice_win": float(np.mean([e["won"] for e in ro["practice_now"]])) if ro["practice_now"] else None,
                "positions_per_s": round(s["n"] / max(s["s"], 1e-9), 1), "playouts": s.get("playouts"), "playout_moves": s.get("playout_moves"),
                "times": {k: round(v, 1) for k, v in s.get("times", {}).items()}}
        with open(os.path.join(a.out, "stats.jsonl"), "a") as f:
            f.write(json.dumps(stat) + "\n")
        print(json.dumps(stat), flush=True)


if __name__ == "__main__":
    main()
