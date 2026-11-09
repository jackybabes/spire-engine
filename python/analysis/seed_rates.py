"""What a seed is worth in whole runs: a checkpoint plays greedy runs with each seed forced, and unseeded.

    .venv/bin/python python/analysis/seed_rates.py python/runs/silent-runs-5/update_1200.pt --gpu-gb 3.5
    .venv/bin/python python/analysis/seed_rates.py python/runs/silent-runs-5/update_1200.pt --fighter python/runs/silent-allstyles/update_600.pt

A seed is free cards, so a seeded run beats an unseeded one for that alone: the control seed (two good
cards that are no engine, crates/sts-py SILENT_SEEDS) is the fair yardstick, and each line gives its
Act 3 reach and its wins against the control's (without a control in --seeds, against the unseeded
runs'). "Won" means both Act 3 bosses at A20; read Act 3 and won, not floors. Every line plays the
same --runs seeds (EVAL_SEED + --seed-offset + j), and the standard error of a ratio resamples those
seeds. "Added" is the engine's cards the run took on top of the seed, up to 6 (`synergy_picks`).

A checkpoint plays as in search/search_eval.py: a deck network (args.fight_net) chooses, with its lookup
if it trained with one, and its fight network fights. --fighter fights instead, for a plain run
network too (unlike search_eval's, which needs args.fight_net), while the lookup stays the fight
network's it trained with. The checkpoint must be what makes the choices: a fight network trained with
--deck-net is refused (its deck network chose, not it; give that as the checkpoint and this as
--fighter), and one that only played practice fights gets a warning. One process: about 3-4 minutes
of the training GPU a line at 512 runs, 20 minutes for the six lines.

--cards gives lines of free cards instead of named seeds, ";" between lines, "," within one ("Catalyst;Accuracy+,Blade Dance"): what a
card is worth to this network, cards it can't play worth little. --free-keys starts every run with the three keys, so one that beats the Act 3 bosses goes
on to Act 4 and the Heart: what its decks would do there, the keys' cost aside (no Recall spent, no chest relic given up, no burning elite).
--seed-act 2 gives the cards on Act 2's first map screen instead of at the start (jack, 2026-11-06: "you need to take some good damage in first act, then build deck to
do poison/discard/other silly things to win"; a gift at floor 1 is mostly early strength). Act 1 is then the same on every line, seed by seed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import json
import time

import numpy as np
import torch

from core.fight_model import pick_device
from core.lookup import Lookup, trained_events
from core.split import SplitPolicy
from core.sts_env import RunEnv
from search.search_eval import beat_act3, play
from training.train_runs import EVAL_SEED, load, load_fight_net

CONTROL = "control"


def load_policy(path, device, amp, fighter=None):
    """(what plays, its lookup or None, the checkpoint): see above."""
    net, ck = load(path, device)
    if ck["args"].get("deck_net"):
        sys.exit(f"{path} trained with --deck-net: its deck network made its choices. Give that as the checkpoint and this as --fighter")
    if ck["args"].get("practice", 0.0) >= 1.0:
        print(f"WARNING: {path} trained on practice fights only, so not on its choices between fights", flush=True)
    net.load_state_dict(ck["net"])
    net.eval()
    trained_with, lookup = ck["args"].get("fight_net"), None
    if trained_with and ck["args"].get("lookup"):
        lookup = Lookup(load_fight_net(trained_with, device), device, amp, draws=ck["args"].get("lookup_draws", 4),
                        every=ck["args"].get("lookup_every", False), events=trained_events(ck["net"]))
    if fighter or trained_with:
        return SplitPolicy(load_fight_net(fighter or trained_with, device), net), lookup, ck
    return net, lookup, ck


def seeded_runs(policy, lookup, device, amp, name, runs, batch, seed, ascension, character, cards=None, free_keys=False, seed_act=1, key_rule=False):
    """`runs` greedy runs with seed `name` forced (None: unseeded; with `cards`, those cards instead), in order of their seeds."""
    b = max(min(batch, runs), 1)
    env = RunEnv(b, seed=seed, ascension=ascension, max_steps=8000, character=character, seeded=1.0 if name else 0.0)
    if cards:
        env.envs.set_seed_cards(cards)
    elif name:
        env.envs.set_seeds([name])
    if free_keys:
        env.envs.set_free_keys(True)
    if key_rule:
        env.envs.set_key_rule(True)
    if seed_act != 1:
        env.envs.set_seed_act(seed_act)
    if name or free_keys or key_rule:
        env.envs.reset()  # these hold from the next episode, and the first ones are already dealt
    out, _ = play(policy, device, amp, env, lambda g, e: e * b + g < runs, None, 0.0, name or "unseeded", lookup=lookup)
    assert all(e["seeded"] == ("cards" if cards else name) for e in out), f"{name}: runs of another seed"
    assert not free_keys or all(e["keys"] >= 3 for e in out), f"{name}: a run without the keys"
    return sorted(out, key=lambda e: e["seed"])


def rate(x):
    return f"{x.mean():6.1%} {x.std() / max(len(x) ** 0.5, 1):5.1%}"


def ratio(x, control, rng, draws=2000):
    """x's mean over control's, with its standard error from resampling the runs (the same seeds in both)."""
    if not control.any():
        return "      -     "
    picks = rng.integers(0, len(x), (draws, len(x)))
    tops, bottoms = x[picks].mean(1), control[picks].mean(1)
    kept = bottoms > 0
    return f"{x.mean() / control.mean():5.2f} ± {(tops[kept] / bottoms[kept]).std():4.2f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--seeds", default="control,poison,shiv,discard,dex", help="the seeds to force, comma-separated (crates/sts-py SILENT_SEEDS...)")
    p.add_argument("--runs", type=int, default=512, help="runs a line")
    p.add_argument("--seed-offset", type=int, default=1024,
                   help="run j has seed EVAL_SEED + this + j (default fresh seeds: no checkpoint was chosen on them; a second set: 8192)")
    p.add_argument("--ascension", type=int, default=20)
    p.add_argument("--batch", type=int, default=256, help="runs at once")
    p.add_argument("--fighter", default=None, help="this checkpoint plays the fights")
    p.add_argument("--cards", default=None, help="lines of free cards instead of named seeds: \"Catalyst;Accuracy+,Blade Dance\"; the control and unseeded lines come first if --seeds holds control")
    p.add_argument("--free-keys", action="store_true", help="every run starts with the three keys")
    p.add_argument("--key-rule", action="store_true", help="for a network trained with keys free: a hand-written rule gets the real keys in Act 3")
    p.add_argument("--seed-act", type=int, default=1, help="the act whose first map screen gives the seed's cards (1: at the start)")
    p.add_argument("--gpu-gb", type=float, default=3.5, help="GPU memory cap, GB (PyTorch's own, without the CUDA context's ~0.45 GB; 0 = none)")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="every run's result as JSON")
    args = p.parse_args()
    device = pick_device(args.device)
    if device.type == "cuda" and args.gpu_gb:
        index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_per_process_memory_fraction(min(args.gpu_gb * 2 ** 30 / torch.cuda.get_device_properties(index).total_memory, 1.0), index)

    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    policy, lookup, ck = load_policy(args.checkpoint, device, amp, args.fighter)
    character = ck["args"].get("character", "ironclad")
    seeds = args.seeds.split(",")
    # The yardstick first, so each line can be read against it as it ends.
    lines = [CONTROL, None] + [s for s in seeds if s != CONTROL] if CONTROL in seeds else [None] + seeds
    custom = {c: c.split(",") for c in args.cards.split(";")} if args.cards else {}
    lines += list(custom)
    print(f"{args.checkpoint} (update {ck.get('update')}), {character} A{args.ascension}, fights by {args.fighter or ck['args'].get('fight_net') or 'itself'}; "
          f"{args.runs} runs a line, seeds {EVAL_SEED + args.seed_offset}..., against {'the control' if CONTROL in seeds else 'unseeded'}"
          f"{'; the three keys free' if args.free_keys else ''}{'; the keys by the hand-written rule' if args.key_rule else ''}{f'; cards given in Act {args.seed_act}' if args.seed_act != 1 else ''}", flush=True)
    print(f"{'seed':10} {'won   ± SE':13} {'Heart ± SE':13} {'Act 2':>5} {'Act 3 ± SE':13} {'floor':>5} {'Act 3 ratio':12} {'won ratio':12} {'added':>5} {'s':>5}", flush=True)
    rng = np.random.default_rng(0)
    results = {}
    against = None
    for name in lines:
        t0 = time.time()
        out = seeded_runs(policy, lookup, device, amp, name, args.runs, args.batch, EVAL_SEED + args.seed_offset, args.ascension, character, custom.get(name), args.free_keys, args.seed_act, args.key_rule)
        act3 = np.array([e["act"] >= 3 for e in out], dtype=float)
        won = np.array([beat_act3(e) for e in out], dtype=float)
        heart = np.array([e["won"] and e["act"] >= 4 for e in out], dtype=float)
        if against is None:
            against = (act3, won)
        print(f"{name or 'unseeded':10} {rate(won)} {rate(heart)} {np.mean([e['act'] >= 2 for e in out]):5.0%} {rate(act3)} "
              f"{np.mean([e['floor'] for e in out]):5.1f} {ratio(act3, against[0], rng):12} {ratio(won, against[1], rng):12} "
              f"{np.mean([e.get('synergy_picks', 0) for e in out]):5.2f} {time.time() - t0:5.0f}", flush=True)
        results[name or "unseeded"] = [{k: e.get(k) for k in ("seed", "won", "act", "floor", "died_in", "keys", "synergy_picks", "archetype_cards")} for e in out]
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"checkpoint": args.checkpoint, "fighter": args.fighter, "seed_offset": args.seed_offset, "ascension": args.ascension, "runs": results}, f)


if __name__ == "__main__":
    main()
