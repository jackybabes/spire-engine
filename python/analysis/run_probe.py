"""Probes what a run checkpoint does and what it would win by doing otherwise.

    .venv/bin/python python/analysis/run_probe.py grids  <run checkpoint>
    .venv/bin/python python/analysis/run_probe.py remove <run checkpoint>
    .venv/bin/python python/analysis/run_probe.py engine <run checkpoint> [--cards "Corruption,Feel No Pain,..."]
    .venv/bin/python python/analysis/run_probe.py sample <run checkpoint> [--runs 512] [--seed 910000] [--out runs.jsonl]

grids:  on the 512 fixed eval runs (greedy), the cards it picks on each kind of grid (upgrade,
        remove, transform, ...), and for removal how often it takes a Strike, Defend or curse.
remove: the same runs with card removal at shops by rule (--when: every shop, with a curse in the
        deck, with N gold to spare; --pick: the network or a curse/Strike/Defend rule picks the
        card), against its own play. (ironclad-runs-4 at update 1750: 81% own play; every shop 46% with its
        own picks, it removed its best cards; 67% removing curses, then Strikes.)
engine: the same runs, made to take an engine card whenever a card reward offers one (default:
        the exhaust, block and strength engines), against its own play.
sample: plays runs with sampled moves, as in training, and reports the strategies: the cards in
        the final decks, card reward pick rates, decks by engine, how alike the decks are, relics,
        boss relic picks, keys, and the fights that ended lost runs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import collections
import json
import random

import numpy as np
import torch

from core.fight_model import masked, pick_device
from analysis.run_choices import CONFIRM, KINDS, PICK, SCREENS, describe, play
from core.run_model import RUN_INT_AT
from core.sts_env import RunEnv
from training.train_runs import load

GRID, SHOP, CARD_REWARD, COMBAT = (SCREENS.index(s) for s in ("grid", "shop", "card_reward", "combat"))
PURGE = KINDS.index("shop_purge")
PICK_CARD = KINDS.index("pick_card")
GRID_KINDS = {KINDS.index(k) for k in KINDS if k.startswith("grid_")}
CURSES = {"Regret", "Shame", "Doubt", "Pain", "Parasite", "Writhe", "Decay", "Normality", "Injury", "Clumsy",
          "Necronomicurse", "Curse of the Bell", "Ascender's Bane", "Pride"}
BASIC = {"Strike", "Defend", "Bash"}
ENGINES = {
    "exhaust": ["Corruption", "Feel No Pain", "Dark Embrace"],
    "block": ["Barricade", "Entrench", "Body Slam", "Juggernaut"],
    "strength": ["Demon Form", "Limit Break", "Inflame", "Spot Weakness"],
}


def base(name):
    return name.rstrip("+")


def view(env, g):
    return json.loads(env.envs.observe(g))


def grids(net, device):
    picks = collections.defaultdict(collections.Counter)

    def on_step(obs, acts, live, env):
        for g in np.flatnonzero(live & (obs["i_screen"][:, 0] == GRID)):
            a = acts[g]
            if PICK <= a < CONFIRM:
                v = view(env, g)
                d = v["screen"]["options"][a - PICK]
                purpose = v["screen"]["purpose"]
                if isinstance(purpose, dict):  # Bottle(relic)
                    purpose = next(iter(purpose))
                picks[purpose][v["deck"][d]["name"]] += 1

    play(net, device, on_step=on_step)
    for purpose, c in sorted(picks.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        line = f"{purpose:16} {n:5} picks: " + ", ".join(f"{k} {v}" for k, v in c.most_common(8))
        if purpose in ("Remove", "ShopPurge", "Toke", "RemoveAndUpgrade"):
            bad = sum(v for k, v in c.items() if base(k) in CURSES | {"Strike", "Defend"})
            line += f"  [curse/Strike/Defend {bad / n:.0%}]"
        print(line)


def remove(net, device, whens, picks):
    """Removal at shops by rule: `whens` are "every" (every shop), "curse" (a curse in the deck),
    "spare:N" (N gold left after paying) or "curse-or-spare:N"; `picks` are "network" (it picks
    the card), "curse-strike" or "curse-defend" (a curse, then a Strike or a Defend, then the
    other). Each combination plays the fixed runs once, against its own play."""
    print("own play:", describe(play(net, device)), flush=True)
    order = {"curse-strike": ("Strike", "Defend"), "curse-defend": ("Defend", "Strike")}
    for when in whens:
        for pick in picks:
            removed, purging, decided = collections.Counter(), {}, {}

            def wanted(v):
                deck = [base(c["name"]) for c in v["deck"]]
                spare = v["gold"] - int(v["screen"].get("purge_cost", 75))
                curse = any(c in CURSES for c in deck)
                if when == "every":
                    return True
                if when == "curse":
                    return curse
                n = int(when.split(":")[1])
                return spare >= n or (when.startswith("curse-or") and curse)

            def adjust(obs, acts, live, env):
                for g in np.flatnonzero(live):
                    sc = obs["i_screen"][g, 0]
                    if sc == SHOP:
                        k = np.flatnonzero(obs["i_opt_kind"][g] == PURGE)
                        if k.size and obs["mask"][g, PICK + k[0]]:
                            v = view(env, g)
                            if decided.get(g) != v["floor"]:
                                decided[g] = v["floor"]
                                if wanted(v):
                                    acts[g] = PICK + k[0]
                            if acts[g] == PICK + k[0]:
                                purging[g] = True
                    elif sc == GRID and purging.pop(g, False):
                        v = view(env, g)
                        names = [v["deck"][d]["name"] for d in v["screen"]["options"]]
                        if pick != "network":
                            first, second = order[pick]
                            for want in (lambda n: base(n) in CURSES, lambda n: base(n) == first, lambda n: base(n) == second):
                                j = next((j for j, n in enumerate(names) if want(n) and obs["mask"][g, PICK + j]), None)
                                if j is not None:
                                    acts[g] = PICK + j
                                    break
                        if PICK <= acts[g] < CONFIRM:
                            removed[names[acts[g] - PICK]] += 1
                    else:
                        purging.pop(g, None)

            res = play(net, device, adjust=adjust)
            print(f"removal when {when}, card by {pick}: {describe(res)}; {sum(removed.values())} removals, "
                  + ", ".join(f"{k} {v}" for k, v in removed.most_common(6)), flush=True)


def engine(net, device, cards):
    print("own play:", describe(play(net, device)), flush=True)
    forced = collections.Counter()

    def adjust(obs, acts, live, env):
        for g in np.flatnonzero(live & (obs["i_screen"][:, 0] == CARD_REWARD)):
            names = [c["name"] for c in view(env, g)["screen"]["cards"]]
            opts = [j for j in range(len(names)) if obs["i_opt_kind"][g, j] == PICK_CARD and obs["mask"][g, PICK + j]]
            for want in cards:
                j = next((j for j in opts if base(names[j]) == want), None)
                if j is not None:
                    forced[want] += acts[g] != PICK + j
                    acts[g] = PICK + j
                    break

    res = play(net, device, adjust=adjust)
    print(f"taking {', '.join(cards)} whenever offered: {describe(res)}; picks it wouldn't have made: "
          + ", ".join(f"{k} {v}" for k, v in forced.most_common()), flush=True)


def sample(net, device, n, seed, out):
    """Plays n runs with sampled moves and records each one from the JSON view."""
    torch.manual_seed(seed)
    env = RunEnv(n, seed=seed, max_steps=8000)
    runs = [dict(card_rewards=[], boss=[], fights=[], shops=0, snap=None) for _ in range(n)]
    done, pending, was_combat = [None] * n, [None] * n, [False] * n
    while any(d is None for d in done):
        obs = env.observe()
        for g in range(n):
            if done[g] is not None:
                continue
            if obs["i_screen"][g, 0] == COMBAT:
                if not was_combat[g]:
                    v = view(env, g)
                    runs[g]["fights"].append((int(v["floor"]), [m.get("name") for m in v["screen"]["combat"]["monsters"]]))
                was_combat[g] = True
                continue
            was_combat[g] = False
            v = view(env, g)
            deck = [c["name"] for c in v["deck"]]
            relics = [r if isinstance(r, str) else r.get("name") for r in v["relics"]]
            if pending[g]:
                kind, offered, before = pending[g]
                if kind == "cards":
                    added = [c for c in deck if deck.count(c) > before.count(c)]
                    runs[g]["card_rewards"].append((int(v["floor"]), offered, added[0] if added else None))
                else:
                    added = [r for r in relics if r not in before]
                    runs[g]["boss"].append((offered, added[0] if added else None))
                pending[g] = None
            sc = v["screen"]
            if sc["type"] == "card_reward":
                pending[g] = ("cards", [c["name"] for c in sc["cards"]], deck)
            elif sc["type"] == "boss_relic":
                pending[g] = ("boss", sc.get("relics", []), relics)
            elif sc["type"] == "shop":
                runs[g]["shops"] += 1
            runs[g]["snap"] = dict(deck=deck, relics=relics, floor=int(v["floor"]), keys=v["keys"])
        with torch.no_grad():
            lg, _ = net(torch.from_numpy(obs["ints"]).to(device).long(), torch.from_numpy(obs["floats"]).to(device))
        acts = torch.distributions.Categorical(logits=masked(lg.float(), torch.from_numpy(obs["mask"]).to(device))).sample()
        for g in np.flatnonzero(env.step(acts.cpu().numpy())):
            if done[g] is None:
                done[g] = json.loads(env.last_episode(int(g)))
    for g in range(n):
        runs[g]["result"] = done[g]
    if out:
        with open(out, "w") as f:
            for r in runs:
                f.write(json.dumps(r) + "\n")
    report(runs)


def report(R):
    n = len(R)
    won = [r for r in R if r["result"]["won"]]
    lost = [r for r in R if not r["result"]["won"]]
    print(f"{n} runs (sampled moves): won {len(won) / n:.1%}, mean floor {np.mean([r['result']['floor'] for r in R]):.1f}")

    def deck(r):
        return r["snap"]["deck"]

    def nonbasic(r):
        return [base(c) for c in deck(r) if base(c) not in BASIC | CURSES]

    print("\nDECKS at the end")
    for name, rs in (("wins", won), ("losses", lost)):
        if rs:
            avg = lambda f: np.mean([f(r) for r in rs])
            print(f"  {name}: {avg(lambda r: len(deck(r))):.1f} cards, Strikes {avg(lambda r: sum(base(c) == 'Strike' for c in deck(r))):.1f}, "
                  f"Defends {avg(lambda r: sum(base(c) == 'Defend' for c in deck(r))):.1f}, upgraded {avg(lambda r: sum(c.endswith('+') for c in deck(r))):.1f}, "
                  f"curses {avg(lambda r: sum(base(c) in CURSES for c in deck(r))):.1f}")
    held, copies = collections.Counter(), collections.Counter()
    for r in R:
        held.update(set(nonbasic(r)))
        copies.update(nonbasic(r))
    print("  most common (share of decks, copies when held, share of wins / losses):")
    for c, k in held.most_common(20):
        w = np.mean([c in nonbasic(r) for r in won]) if won else 0
        l = np.mean([c in nonbasic(r) for r in lost]) if lost else 0
        print(f"    {c:18} {k / n:4.0%}  x{copies[c] / k:.1f}  ({w:.0%} / {l:.0%})")

    offered, taken, total, skipped = collections.Counter(), collections.Counter(), 0, 0
    for r in R:
        for _, offer, chosen in r["card_rewards"]:
            total += 1
            skipped += chosen is None
            offered.update(set(map(base, offer)))
            if chosen:
                taken[base(chosen)] += 1
    rates = sorted(((taken[c] / offered[c], c) for c in offered if offered[c] >= 25), reverse=True)
    print(f"\nCARD REWARDS: {total}, skipped {skipped / max(total, 1):.0%}")
    print("  most taken:  " + ", ".join(f"{c} {p:.0%}" for p, c in rates[:15]))
    print("  least taken: " + ", ".join(f"{c} {p:.0%}" for p, c in rates[-15:]))

    print("\nENGINES (two or more core cards in the final deck)")
    groups, group_wins = collections.Counter(), collections.Counter()
    for r in R:
        s = set(nonbasic(r))
        key = "+".join(k for k, cards in ENGINES.items() if len(s & set(cards)) >= 2) or "none"
        groups[key] += 1
        group_wins[key] += r["result"]["won"]
    for k, v in groups.most_common():
        print(f"  {k:24} {v / n:4.0%} of runs, won {group_wins[k] / v:.0%}")
    for k, cards in ENGINES.items():
        print(f"  {k:9} cards held: " + ", ".join(f"{c} {np.mean([c in nonbasic(r) for r in R]):.0%}" for c in cards))

    sets = [set(nonbasic(r)) for r in R]
    rng = random.Random(0)
    pairs = [rng.sample(range(n), 2) for _ in range(2000)]
    sims = [len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in pairs if sets[a] | sets[b]]
    print(f"\nHOW ALIKE: two decks share {np.mean(sims):.0%} of their non-basic cards (Jaccard, 2000 pairs); "
          f"in 3 of 4 decks: {[c for c, k in held.items() if k / n >= 0.75]}")

    relics = collections.Counter(x for r in R for x in set(r["snap"]["relics"]) if x not in ("Burning Blood", "NeowsBlessing"))
    print("\nRELICS held: " + ", ".join(f"{c} {k / n:.0%}" for c, k in relics.most_common(12)))
    boss_offered, boss_taken = collections.Counter(), collections.Counter()
    for r in R:
        for offer, chosen in r["boss"]:
            boss_offered.update(offer)
            if chosen:
                boss_taken[chosen] += 1
    ranked = sorted((c for c in boss_offered if boss_offered[c] >= 15), key=lambda c: -boss_taken[c] / boss_offered[c])
    print("BOSS RELICS taken when offered: " + ", ".join(f"{c} {boss_taken[c] / boss_offered[c]:.0%}" for c in ranked))
    keys = {k: np.mean([r["snap"]["keys"][k] for r in R]) for k in ("emerald", "ruby", "sapphire")}
    print("KEYS: " + ", ".join(f"{k} {v:.0%}" for k, v in keys.items()))
    ended = collections.Counter("+".join(sorted(set(r["fights"][-1][1]))) if r["fights"] else "?" for r in lost)
    print("LOST RUNS ended at: " + ", ".join(f"{k} {v}" for k, v in ended.most_common(10)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("probe", choices=["grids", "remove", "engine", "sample"])
    p.add_argument("checkpoint")
    p.add_argument("--cards", default=",".join(c for cards in ENGINES.values() for c in cards))
    p.add_argument("--when", default="every", help="remove: when to remove (comma-separated: every, curse, spare:N, curse-or-spare:N)")
    p.add_argument("--pick", default="network,curse-strike", help="remove: who picks the card (network, curse-strike, curse-defend)")
    p.add_argument("--runs", type=int, default=512)
    p.add_argument("--seed", type=int, default=910_000)
    p.add_argument("--out")
    args = p.parse_args()
    device = pick_device()
    net, ck = load(args.checkpoint, device)
    net.load_state_dict(ck["net"])
    net.eval()
    print(f"{args.checkpoint} (update {ck['update']})")
    if args.probe == "grids":
        grids(net, device)
    elif args.probe == "remove":
        remove(net, device, args.when.split(","), args.pick.split(","))
    elif args.probe == "engine":
        engine(net, device, [c.strip() for c in args.cards.split(",") if c.strip()])
    else:
        sample(net, device, args.runs, args.seed, args.out)


if __name__ == "__main__":
    main()
