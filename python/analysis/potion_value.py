"""What a potion is worth in HP (jack, 2026-10-30: "How much hp is saved on average when a potion is used? That is
a metric humans use when playing. Potions = hp sorta?"), measured the only way that isn't fooled by potions being
drunk in hard fights: the same fight played with the potion and without it.

    python python/analysis/potion_value.py python/runs/ironclad-runs-habits-12/update_200.pt \\
        --scenarios ~/sts-data/bosses-ironclad/eval.jsonl.gz --k 4

Every held-out human fight (human_bosses.py's scenarios: the deck, relics, HP and potions a winner brought) that
holds a potion is played --k times greedily by the network as it was, --k times without each of its potions in
turn, and --k times with none. HP left counts 0 for a lost fight. A potion's worth in one fight is the mean HP left
with all the potions minus the mean without that one; averaged by potion and by kind of fight (hallway fights
aren't in the human set: Act 1-3 bosses, the Heart, Act 1-3 elites), with the win rate's change. --out keeps every
fight's result as JSON lines.
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

from core.fight_model import masked, pick_device  # noqa: E402
from core.sts_env import RunEnv  # noqa: E402
from training.train_runs import load, read_scenarios  # noqa: E402

SEED = 1_100_000


def play(net, device, amp, lines, character, seed, batch):
    """Each scenario line played once, greedily: (won, HP left, max HP, potions left)."""
    out = []
    for at in range(0, len(lines), batch):
        chunk = lines[at:at + batch]
        n = len(chunk)
        env = RunEnv(n, seed=seed + at, practice=1.0, max_steps=4000, character=character)
        skipped = env.envs.set_scenarios(chunk, 1.0, True)
        if skipped:
            raise ValueError(f"{skipped} scenarios couldn't be set up")
        res = [None] * n
        prac = torch.ones(n, dtype=torch.bool, device=device)
        while any(r is None for r in res):
            obs = env.observe()
            ints = torch.from_numpy(obs["ints"]).to(device).long()
            floats = torch.from_numpy(obs["floats"]).to(device)
            mask = torch.from_numpy(obs["mask"]).to(device)
            with torch.no_grad(), amp():
                logits, _ = net(ints, floats, prac)
            status = env.step(masked(logits.float(), mask).argmax(-1).cpu().numpy())
            for g in np.flatnonzero(status):
                if res[g] is None:
                    e = json.loads(env.last_episode(int(g)))
                    res[g] = (bool(e["won"]), e["hp"] if e["won"] else 0, e["max_hp"], e.get("potions", 0))
        out += res
        print(f"  {at + n:,} of {len(lines):,} fights", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--scenarios", required=True)
    p.add_argument("--character", default="ironclad", choices=["ironclad", "silent", "defect", "watcher"])
    p.add_argument("--k", type=int, default=4, help="copies of each variant of a fight")
    p.add_argument("--batch", type=int, default=2048, help="fights played at once")
    p.add_argument("--limit", type=int, default=None, help="only the first N fights with potions (a quick look)")
    p.add_argument("--out", help="every fight's result as JSON lines")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = pick_device(args.device)
    amp = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)) if device.type == "cuda" else (lambda: torch.autocast("cpu", enabled=False))
    net, ck = load(args.checkpoint, device)
    net.load_state_dict(ck["net"])
    net.eval()

    fights = [json.loads(line) for line in read_scenarios(os.path.expanduser(args.scenarios))]
    fights = [f for f in fights if f.get("potions")]
    if args.limit:
        fights = fights[:args.limit]
    # The variants: as it was (None), without potion slot i (i), without any ("none"); --k copies of each.
    lines, meta = [], []
    for fi, f in enumerate(fights):
        variants = [None] + list(range(len(f["potions"]))) + (["none"] if len(f["potions"]) > 1 else [])
        for v in variants:
            pots = f["potions"] if v is None else [] if v == "none" else f["potions"][:v] + f["potions"][v + 1:]
            for c in range(args.k):
                lines.append(json.dumps({**f, "potions": pots}))
                meta.append((fi, v, c))
    t0 = time.time()
    print(f"{len(fights):,} human fights with potions, {len(lines):,} fights to play", flush=True)
    res = play(net, device, amp, lines, args.character, SEED, args.batch)
    print(f"played in {time.time() - t0:.0f} s", flush=True)

    by = collections.defaultdict(list)  # (fight, variant) -> results
    for (fi, v, c), r in zip(meta, res):
        by[(fi, v)].append(r)
    if args.out:
        with open(args.out, "w") as fo:
            for (fi, v, c), r in zip(meta, res):
                fo.write(json.dumps({"fight": fi, "kind": f"{fights[fi].get('room', 'boss')}{fights[fi]['act']}",
                                     "potions": fights[fi]["potions"], "without": v, "won": r[0], "hp": r[1], "max_hp": r[2],
                                     "potions_left": r[3]}) + "\n")

    def kind(f):
        return "the Heart" if f["act"] == 4 and f.get("room", "boss") == "boss" else f"Act {f['act']} {f.get('room', 'boss')}"

    per_potion = collections.defaultdict(lambda: {"hp": [], "win": [], "hp_pct": []})
    per_kind = collections.defaultdict(lambda: {"hp": [], "win": [], "hp_pct": []})
    all_none = collections.defaultdict(lambda: {"hp": [], "win": []})
    for fi, f in enumerate(fights):
        base = by[(fi, None)]
        hp0, win0 = np.mean([r[1] for r in base]), np.mean([r[0] for r in base])
        mx = base[0][2]
        for i, pot in enumerate(f["potions"]):
            wo = by[(fi, i)]
            d_hp, d_win = hp0 - np.mean([r[1] for r in wo]), win0 - np.mean([r[0] for r in wo])
            for key, table in ((pot, per_potion), (kind(f), per_kind)):
                table[key]["hp"].append(d_hp)
                table[key]["win"].append(d_win)
                table[key]["hp_pct"].append(100 * d_hp / mx)
        if (fi, "none") in by:
            no = by[(fi, "none")]
            all_none[kind(f)]["hp"].append(hp0 - np.mean([r[1] for r in no]))
            all_none[kind(f)]["win"].append(win0 - np.mean([r[0] for r in no]))

    def line(name, d):
        hp = np.array(d["hp"])
        return (f"  {name:28} {len(hp):5}  HP saved {hp.mean():5.1f} (±{hp.std(ddof=1) / np.sqrt(len(hp)) if len(hp) > 1 else 0:.1f}), "
                f"{np.mean(d['hp_pct']):4.1f}% of max HP, win {100 * np.mean(d['win']):+5.1f} points")

    print("\nOne potion, by kind of fight (fights, HP saved by that potion, its share of max HP, the win rate's change):")
    for k in sorted(per_kind, key=lambda k: -np.mean(per_kind[k]["hp"])):
        print(line(k, per_kind[k]))
    print("\nBy potion (at least 20 fights):")
    for pot in sorted(per_potion, key=lambda k: -np.mean(per_potion[k]["hp"])):
        if len(per_potion[pot]["hp"]) >= 20:
            print(line(pot, per_potion[pot]))
    every = {"hp": [x for d in per_potion.values() for x in d["hp"]], "win": [x for d in per_potion.values() for x in d["win"]],
             "hp_pct": [x for d in per_potion.values() for x in d["hp_pct"]]}
    print("\n" + line("Any one potion", every).strip())
    print("\nAll the potions against none (fights holding two or more):")
    for k in sorted(all_none):
        hp = np.array(all_none[k]["hp"])
        print(f"  {k:28} {len(hp):5}  HP saved {hp.mean():5.1f}, win {100 * np.mean(all_none[k]['win']):+5.1f} points")


if __name__ == "__main__":
    main()
