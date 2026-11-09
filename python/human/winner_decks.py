"""Which cards human winners' decks hold, and which they hold together.

    python3 python/human/winner_decks.py [~/sts-data/watcher] [--asc 20] [--all-wins] [--top 70] [--out decks.json]

Reads a character's runs (`human_extract.py`'s <out>/runs) at one ascension and, for the runs that
beat the Heart (won at floor 55 or more; --all-wins: any win), prints the deck size, upgrades,
relics and Strikes and Defends left (quartiles), then each card's share of those decks against its
share of all the ascension's decks (the lift), its copies per deck, and how often those winners
took it when offered; then the pairs of cards held together most beyond chance (the share
of decks with both over the product of their shares). Cards are the dump's ids (the game's
old ones: Adaptation is Rushdown, Fasting2 Fasting, Wireheading Foresight, ClearTheMind
Tranquility, Vengeance Simmering Fury). Chose the Watcher's practice engines (2026-10-27,
`crates/sts-py/src/practice.rs`). For what a network knows of combinations, see
`python/analysis/synergy.py`.
"""
import argparse
import glob
import gzip
import itertools
import json
import os
from collections import Counter

HEART_FLOOR = 55
STARTERS = {"Strike_P", "Defend_P", "Eruption", "Vigilance", "Strike_R", "Defend_R", "Bash", "Strike_G", "Defend_G", "Neutralize",
            "Survivor", "Strike_B", "Defend_B", "Zap", "Dualcast", "AscendersBane"}


def base(card):
    return card.split("+")[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", nargs="?", default="~/sts-data/watcher")
    p.add_argument("--asc", type=int, default=20)
    p.add_argument("--all-wins", action="store_true", help="any win, not only the Heart")
    p.add_argument("--top", type=int, default=70)
    p.add_argument("--pair-min", type=float, default=0.02, help="cards in at least this share of the wins are paired")
    p.add_argument("--out")
    args = p.parse_args()

    pres_win, pres_all, copies_win, offered, taken = Counter(), Counter(), Counter(), Counter(), Counter()
    n_all = 0
    stats = {"deck": [], "upgraded": [], "relics": [], "basics": []}
    decks = []
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(args.data), "runs", "*.jsonl.gz"))):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            if r.get("ascension_level") != args.asc:
                continue
            deck = [base(c) for c in r.get("master_deck", [])]
            n_all += 1
            pres_all.update(set(deck))
            if not r.get("victory") or (not args.all_wins and r.get("floor_reached", 0) < HEART_FLOOR):
                continue
            pres_win.update(set(deck))
            copies_win.update(deck)
            decks.append(set(deck))
            stats["deck"].append(len(deck))
            stats["upgraded"].append(sum("+" in c for c in r["master_deck"]))
            stats["relics"].append(len(r.get("relics", [])))
            stats["basics"].append(sum(c.startswith(("Strike_", "Defend_")) for c in deck))
            for ch in r.get("card_choices", []):
                for c in ch.get("not_picked", []):
                    offered[base(c)] += 1
                picked = ch.get("picked")
                if picked and picked not in ("SKIP", "Singing Bowl"):
                    offered[base(picked)] += 1
                    taken[base(picked)] += 1
    n_win = len(decks)
    if not n_win:
        raise SystemExit(f"no winning runs at A{args.asc} in {args.data}")
    print(f"A{args.asc} runs {n_all:,}, {'wins' if args.all_wins else 'Heart wins'} {n_win:,}")
    for name, xs in stats.items():
        xs = sorted(xs)
        print(f"  {name}: 25% {xs[len(xs) // 4]}, median {xs[len(xs) // 2]}, 75% {xs[3 * len(xs) // 4]}")

    rows = []
    for c, k in pres_win.items():
        if c in STARTERS:
            continue
        pw, pa = k / n_win, pres_all[c] / n_all
        rows.append({"card": c, "in_wins": pw, "in_all": pa, "lift": pw / pa, "copies": copies_win[c] / k,
                     "taken": taken[c] / offered[c] if offered[c] else None, "offered": offered[c]})
    rows.sort(key=lambda x: -x["in_wins"])
    print(f"\n{'card':22} {'in wins':>8} {'in all':>7} {'lift':>5} {'copies':>6} {'taken':>6} {'offered':>8}")
    for r in rows[:args.top]:
        tr = f"{r['taken']:6.1%}" if r["taken"] is not None else "     -"
        print(f"{r['card']:22} {r['in_wins']:8.1%} {r['in_all']:7.1%} {r['lift']:5.2f} {r['copies']:6.2f} {tr} {r['offered']:8}")

    # Pairs: every card in at least --pair-min of the winning decks. A deck's size drives what it
    # holds (a 30-card deck holds more of everything), so the expected count of a pair is summed
    # over deck-size groups: each group's decks times the two cards' shares within it.
    cards = [r["card"] for r in rows if r["in_wins"] >= args.pair_min]
    groups = {}
    for d, n in zip(decks, stats["deck"]):
        groups.setdefault(min(n // 4, 9), []).append(d)
    share_in = {g: {c: sum(c in d for d in ds) / len(ds) for c in cards} for g, ds in groups.items()}
    paired = set(cards)
    together = Counter()
    for d in decks:
        together.update(itertools.combinations(sorted(d & paired), 2))
    share = {c: pres_win[c] / n_win for c in cards}
    pairs = []
    for a, b in itertools.combinations(sorted(cards), 2):
        both = together[(a, b)]
        expected = sum(len(ds) * share_in[g][a] * share_in[g][b] for g, ds in groups.items())
        if expected < 1:
            continue
        pairs.append({"a": a, "b": b, "decks": both, "expected": expected, "lift": both / n_win / (share[a] * share[b]),
                      "size_lift": both / expected})
    print(f"\n{len(cards)} cards in at least {args.pair_min:.0%} of the wins, {len(pairs):,} pairs; lift against deck size "
          f"(a pair's decks over what decks of the same sizes would hold by chance):")
    strong = [x for x in pairs if x["decks"] >= 30]
    strong.sort(key=lambda x: -x["size_lift"])
    print(f"  {sum(x['size_lift'] >= 1.5 for x in strong)} pairs at 1.5 or more, {sum(x['size_lift'] >= 2 for x in strong)} at 2 or more "
          f"(of {len(strong)} held together in 30 decks or more)")
    print("\nheld together most (size lift, plain lift, decks with both / expected):")
    for x in strong[:args.top]:
        print(f"  {x['size_lift']:4.2f} {x['lift']:4.2f}  {x['a']} + {x['b']} ({x['decks']} / {x['expected']:.0f})")
    apart = [x for x in pairs if x["expected"] >= 30]
    apart.sort(key=lambda x: x["size_lift"])
    print("\nheld apart most (cards that rarely share a winning deck):")
    for x in apart[:args.top // 2]:
        print(f"  {x['size_lift']:4.2f} {x['lift']:4.2f}  {x['a']} + {x['b']} ({x['decks']} / {x['expected']:.0f})")
    if args.out:
        json.dump({"asc": args.asc, "runs": n_all, "wins": n_win, "cards": rows, "pairs": pairs}, open(args.out, "w"))


if __name__ == "__main__":
    main()
