"""Can a fight network use a card? On human fights (human_bosses.py's scenario lines: the deck, relics, potions
and HP each winning run brought to a boss or elite, so decks that won, not every player's), greedily:

    .venv/bin/python python/analysis/fights_by_card.py python/runs/silent-allstyles/update_600.pt --gpu-gb 3.5 --out allstyles.json
    .venv/bin/python python/analysis/fights_by_card.py python/runs/silent-runs-5/update_1200.pt --lines 6000 --removal-fights 1000

(a) The win rate of the fights whose deck holds the card, by kind (boss1-4, elite1-3), and the same
    without it. "Won" at boss3 means both bosses of the A20 pair. Whose deck holds what isn't chance, so
    this says where the card is met, not what it does; (c) does that.
(b) What the network does with it. When it is in hand and playable (a legal move plays it): on the share
    of those turns and of those fights that it gets played. Catalyst: the poison on the target when
    played, and the play rate by the most poison on an enemy that turn. Card selections: what is picked
    (curses, Tactician and Reflex, Strikes and Defends...) against picking at random among the
    options, and the share of the times a listed card is on offer that it gets picked. Tactician and
    Reflex can't be played: their row is the discard one.
(c) Removal: each fight holding the card played again with one copy gone (the least upgraded), on the
    same seed: the same monsters, HP and potions, but other draws (a deck one card smaller shuffles
    differently, so the opening hands hardly ever match). The paired difference in wins (with the
    card minus without) is what the card is worth to this network in its fights, with its standard
    error; below zero, the network does better without it. The same in `practice_reward` beside it (a
    win is 50 points, HP left the rest). Strike and Defend are in the default list as the yardstick:
    what removing a plain card does.

One process. --lines takes that many fights evenly through the lines (0: all 59,120), which (a) and (b)
report on; --removal-fights then takes that many of those holding each card, evenly, at about 1.1
points of standard error for 2,000, 1.6 for 1,000. Each card's row prints as it ends, in the order of --cards. Lines
with 30 relics or more are left out (cheated runs; the new filter, notes/silent-replan-2026-11-06.md).
The checkpoint's fight network plays: its own args.fight_net if it has one (a deck network's), else
itself. A network that practised on train.jsonl.gz has met those fights: the first lines compare them
with eval.jsonl.gz's.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import collections
import json
import time

import numpy as np
import torch

import sts
from analysis.archetypes import ORDER, arch
from core.fight_model import HAND_FEATS, MAX_MONSTERS, masked, pick_device
from core.sts_env import LAYOUT, RunEnv
from training.train_runs import SCENARIO_EVAL_SEED, in_combat, load, load_fight_net, practice_reward, read_scenarios

CARDS = ("Catalyst,Noxious Fumes,Bouncing Flask,Deadly Poison,Footwork,Backflip,Acrobatics,Well-Laid Plans,Tactician,Wraith Form,Blade Dance,"
         "Accuracy,Dash,Strike_G,Defend_G")
KINDS = ["boss1", "boss2", "boss3", "boss4", "elite1", "elite2", "elite3", "elite4"]
SCENARIOS = "~/sts-data/bosses-silent3/eval.jsonl.gz,~/sts-data/bosses-silent3/train.jsonl.gz"
MAX_RELICS = 30

POTION = LAYOUT["actions"]["potion"][0]
TARGETS = LAYOUT["actions"]["play"][2]
PICK, MAX_OPTIONS = LAYOUT["actions"]["pick"]
TURN_AT = 5  # encode.rs scalars
PLAYABLE_AT = 5  # encode.rs hand_row
MAX_MONSTER_POWERS = sum(n for f, n in LAYOUT["ints"] if f == "monster_power") // MAX_MONSTERS
POISON_EDGES = [1, 5, 10, 20]
POISON_LABELS = ["0", "1-4", "5-9", "10-19", "20+"]
CLASSES = ["curse or status", "Tactician or Reflex", "Strike or Defend", "attack", "skill", "power"]


def card_class(i):
    """What kind of card encoded id - 1 `i` is, for what a selection picks (CLASSES' index)."""
    java, kind = LAYOUT["card_java_ids"][i], LAYOUT["card_types"][i]
    if kind in ("Curse", "Status"):
        return 0
    if java in ("Tactician", "Reflex"):
        return 1
    if java.startswith(("Strike_", "Defend_")):
        return 2
    return {"Attack": 3, "Skill": 4, "Power": 5}[kind]


CARD_CLASS = np.array([0] + [card_class(i) for i in range(len(LAYOUT["card_names"]))])


def card_ids(names):
    """Each card name or game id (Strike_G: several cards are called Strike) as its index in the layout's card lists."""
    known = {n: i for i, n in enumerate(LAYOUT["card_names"])}
    known.update({j: i for i, j in enumerate(LAYOUT["card_java_ids"])})
    unknown = [n for n in names if n not in known]
    if unknown:
        sys.exit(f"unknown cards: {unknown}")
    return [known[n] for n in names]


def evenly(items, n):
    """`n` of `items`, evenly through them (all of them if n is 0 or more than there are)."""
    return [items[i * len(items) // n] for i in range(n)] if 0 < n < len(items) else list(items)


def kind_of(fight):
    return f"{'elite' if fight.get('room') == 'elite' else 'boss'}{fight['act']}"


def read_fights(paths, character, kinds, lines):
    """The fights to play: each a scenario's JSON with its "seed" (SCENARIO_EVAL_SEED + its place in `paths`, so a fight has
    the same seed however it is sampled, and the first file's seeds are search_eval.py's), and its file as "source"."""
    fights, at, cheated = [], 0, 0
    for path in paths.split(","):
        for line in read_scenarios(os.path.expanduser(path)):
            fight = json.loads(line)
            fight["seed"] = SCENARIO_EVAL_SEED + at
            at += 1
            if len(fight["relics"]) >= MAX_RELICS:
                cheated += 1
            elif not kinds or kind_of(fight) in kinds:
                fight["source"] = os.path.basename(path).split(".")[0]
                fights.append(fight)
    picked = evenly(fights, lines)
    # Lines the engine can't set up: skipped by set_scenarios (which would move every later fight's place) or, once
    # in a sequential batch, an error.
    env = sts.Envs(1, mode="run", practice=1.0, character=character)
    usable = []
    for fight in picked:
        try:
            if env.set_scenarios([json.dumps(fight)], 1.0, True) == 0:
                usable.append(fight)
        except ValueError:
            pass
    print(f"{at} lines, {cheated} with {MAX_RELICS}+ relics left out; {len(picked)} fights taken, {len(picked) - len(usable)} of those can't be set up", flush=True)
    return usable


def without(fight, java):
    """The fight with one copy of the card gone (the least upgraded)."""
    deck = fight["deck"]
    at = min((i for i, c in enumerate(deck) if c[0] == java), key=lambda i: deck[i][1])
    return dict(fight, deck=deck[:at] + deck[at + 1:])


class Plays:
    """(b)'s counts, over a batch of `n` games: `update` before each step, `end` for the games whose fight ended in it.
    `cards`: the layout's indices of the listed cards. Rows outside fights, and fights not `counted`, aren't counted."""

    def __init__(self, cards, n):
        self.cards, self.m, self.n = list(cards), len(cards), n
        self.table = np.full(len(LAYOUT["card_names"]) + 1, self.m)
        self.table[np.array(self.cards) + 1] = np.arange(self.m)
        names = [LAYOUT["card_java_ids"][c] for c in self.cards]
        self.catalyst = names.index("Catalyst") if "Catalyst" in names else None
        # This turn's and this fight's: the card was playable; it was played.
        self.seen_turn, self.played_turn = np.zeros((n, self.m + 1), bool), np.zeros((n, self.m + 1), bool)
        self.seen_fight, self.played_fight = np.zeros((n, self.m + 1), bool), np.zeros((n, self.m + 1), bool)
        self.turn = np.zeros(n)
        self.most_poison = np.zeros(n)
        self.turns, self.turns_played = np.zeros(self.m + 1, dtype=int), np.zeros(self.m + 1, dtype=int)
        self.fights, self.fights_played = np.zeros(self.m + 1, dtype=int), np.zeros(self.m + 1, dtype=int)
        self.plays = np.zeros(self.m + 1, dtype=int)
        self.poison_turns, self.poison_turns_played = np.zeros(len(POISON_LABELS), dtype=int), np.zeros(len(POISON_LABELS), dtype=int)
        self.poison_target = np.zeros(len(POISON_LABELS), dtype=int)
        self.on_most_poison = 0
        # By choice kind: picks and stops; what's picked by class; what picking at random would give; a listed card on offer, and picked.
        self.decisions, self.stops = collections.Counter(), collections.Counter()
        self.picked_class = collections.defaultdict(lambda: np.zeros(len(CLASSES)))
        self.random_class = collections.defaultdict(lambda: np.zeros(len(CLASSES)))
        self.offered, self.taken = collections.Counter(), collections.Counter()

    def close_turns(self, games):
        """Closes the turns of `games` (a bool mask)."""
        self.turns += self.seen_turn[games].sum(0)
        self.turns_played += (self.seen_turn & self.played_turn)[games].sum(0)
        if self.catalyst is not None:
            held = games & self.seen_turn[:, self.catalyst]
            bucket = np.digitize(self.most_poison[held], POISON_EDGES)
            np.add.at(self.poison_turns, bucket, 1)
            np.add.at(self.poison_turns_played, bucket, self.played_turn[held, self.catalyst].astype(int))
        self.seen_turn[games] = False
        self.played_turn[games] = False
        self.most_poison[games] = 0

    def update(self, obs, acts, counted):
        turn = obs["f_scalars"][:, TURN_AT]
        self.close_turns(turn != self.turn)
        self.turn = turn.copy()
        games = np.flatnonzero(counted & in_combat(obs["ints"]))
        if not len(games):
            return
        hand = self.table[obs["i_hand_card"][games]]
        playable = obs["f_hand"][games][:, PLAYABLE_AT::HAND_FEATS] > 0
        r, h = np.nonzero(playable & (hand < self.m))
        self.seen_turn[games[r], hand[r, h]] = True
        self.seen_fight[games[r], hand[r, h]] = True
        moves = acts[games]
        r = np.flatnonzero(moves < POTION)
        cards = hand[r, moves[r] // TARGETS]
        self.played_turn[games[r], cards] = True
        self.played_fight[games[r], cards] = True
        np.add.at(self.plays, cards, 1)
        if self.catalyst is not None:
            poison = (obs["i_monster_power"][games] == LAYOUT["poison_power"]) * obs["f_monster_power_amount"][games]
            poison = poison.reshape(len(games), MAX_MONSTERS, MAX_MONSTER_POWERS).sum(-1)
            can = ((hand == self.catalyst) & playable).any(1)
            self.most_poison[games[can]] = np.maximum(self.most_poison[games[can]], poison[can].max(1))
            for row in r[cards == self.catalyst]:
                on = poison[row, moves[row] % TARGETS - 1]
                self.poison_target[np.digitize(on, POISON_EDGES)] += 1
                self.on_most_poison += on >= poison[row].max()
        for g in games[obs["i_choice_kind"][games, 0] > 0]:
            self.choice(LAYOUT["choice_kinds"][obs["i_choice_kind"][g, 0] - 1], obs["i_option_card"][g], obs["mask"][g, PICK:PICK + MAX_OPTIONS], int(acts[g]))

    def choice(self, kind, options, can, act):
        on_offer = options[can]
        self.decisions[kind] += 1
        if len(on_offer):
            self.random_class[kind] += np.bincount(CARD_CLASS[on_offer], minlength=len(CLASSES)) / len(on_offer)
        pick = options[act - PICK] if PICK <= act < PICK + MAX_OPTIONS else None
        if pick is None:
            self.stops[kind] += 1
        else:
            self.picked_class[kind][CARD_CLASS[pick]] += 1
        for c in set(self.table[on_offer].tolist()) - {self.m}:
            self.offered[kind, c] += 1
            self.taken[kind, c] += pick is not None and self.table[pick] == c

    def end(self, games):
        self.close_turns(games)
        self.fights += self.seen_fight[games].sum(0)
        self.fights_played += (self.seen_fight & self.played_fight)[games].sum(0)
        self.seen_fight[games] = False
        self.played_fight[games] = False

    def report(self, names):
        """The counts as JSON's lists, by card name."""
        rows = {n: {"turns": int(self.turns[i]), "turns_played": int(self.turns_played[i]), "fights": int(self.fights[i]),
                    "fights_played": int(self.fights_played[i]), "plays": int(self.plays[i])} for i, n in enumerate(names)}
        for (kind, i), k in self.offered.items():
            rows[names[i]].setdefault("selections", {})[kind] = {"offered": k, "picked": int(self.taken[kind, i])}
        out = {"cards": rows, "selections": {k: {"decisions": self.decisions[k], "stops": self.stops[k],
                                                 "picked": dict(zip(CLASSES, self.picked_class[k].tolist())),
                                                 "at_random": dict(zip(CLASSES, self.random_class[k].tolist()))} for k in self.decisions}}
        if self.catalyst is not None:
            out["catalyst"] = {"poison": POISON_LABELS, "turns": self.poison_turns.tolist(), "turns_played": self.poison_turns_played.tolist(),
                               "plays_on": self.poison_target.tolist(), "on_most_poison": int(self.on_most_poison)}
        return out


def play_fights(net, device, amp, fights, batch, character, plays=None):
    """Each of `fights` once, greedily, in the fights' own seeds: their episodes, in order. `plays`: counts them (a `Plays` of
    `batch` games, or of the fights if fewer)."""
    if not fights:
        return []
    n = len(fights)
    b = min(batch, n)
    env = RunEnv(b, seed=SCENARIO_EVAL_SEED, practice=1.0, max_steps=4000, character=character)
    assert env.envs.set_scenarios([json.dumps(f) for f in fights], 1.0, True) == 0
    # Game g's episode e plays fight e * b + g (Envs.set_scenarios, sequential); past the last it has wrapped round, and only
    # takes its first legal move (no pass through the network).
    episode, out, games = np.zeros(b, dtype=int), [None] * n, np.arange(b)
    while (episode * b + games < n).any():
        obs = env.observe()
        live = episode * b + games < n
        rows = torch.from_numpy(live).to(device)
        with torch.no_grad(), amp():
            logits, _ = net(torch.from_numpy(obs["ints"]).to(device).long()[rows], torch.from_numpy(obs["floats"]).to(device)[rows],
                            torch.ones(int(live.sum()), dtype=torch.bool, device=device))
        acts = obs["mask"].argmax(1)
        acts[live] = masked(logits.float(), torch.from_numpy(obs["mask"]).to(device)[rows]).argmax(-1).cpu().numpy()
        if plays:
            plays.update(obs, acts, live)
        status = env.step(acts)
        for g in np.flatnonzero(status):
            i = episode[g] * b + g
            if i < n:
                out[i] = json.loads(env.last_episode(int(g)))
                assert out[i]["scenario"] == i and out[i]["practice"] == kind_of(fights[i]), (i, out[i])
            episode[g] += 1
        if plays:
            plays.end(status > 0)
    return out


def mean_se(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std() / max(len(x) ** 0.5, 1))


def cell(x):
    """Win rate ± standard error (fights), in points."""
    if not len(x):
        return f"{'-':>17}"
    m, s = mean_se(x)
    return f"{100 * m:5.1f}±{100 * s:3.1f} ({len(x):5})"


def report_wins(fights, eps, names, held):
    """(a), returned as JSON too."""
    won = np.array([e["won"] for e in eps], dtype=float)
    kinds = np.array([kind_of(f) for f in fights])
    sources = np.array([f["source"] for f in fights])
    print("\nAll the fights: " + ", ".join(f"{k} {cell(won[kinds == k]).strip()}" for k in KINDS if (kinds == k).any()))
    print("By file: " + ", ".join(f"{s} {cell(won[sources == s]).strip()}" for s in dict.fromkeys(sources.tolist())))
    first = [e.get("bosses_beaten", 0) >= 1 for e, k in zip(eps, kinds) if k == "boss3"]
    if first:
        print(f"boss3's first boss: {np.mean(first):.1%}; timed out: {sum(e.get('timed_out', False) for e in eps)}")
    print("\n(a) Won, with its standard error (fights), in the fights whose deck holds the card; then the fights without it")
    used = [k for k in KINDS if (kinds == k).any()]
    print(f"{'card':16} {'':8} " + " ".join(f"{k:>17}" for k in ["all", *used]))
    table = {}
    for name, h in zip(names, held):
        has = np.zeros(len(fights), dtype=bool)
        has[h] = True
        for label, rows in (("holds", has), ("without", ~has)):
            print(f"{name if label == 'holds' else '':16} {label:8} {cell(won[rows])} " + " ".join(cell(won[rows & (kinds == k)]) for k in used))
            table.setdefault(name, {})[label] = {k: [int((rows & (kinds == k)).sum()), float(won[rows & (kinds == k)].sum())] for k in used}
    return table


def report_archetypes(fights, eps):
    """Won by the deck's archetype (analysis/archetypes.py) and kind: can the network carry each kind of deck through this fight? Returned as JSON too."""
    won = np.array([e["won"] for e in eps], dtype=float)
    kinds = np.array([kind_of(f) for f in fights])
    archs = np.array([arch(c[0] for c in f["deck"]) for f in fights])
    used = [k for k in KINDS if (kinds == k).any()]
    print("\nBy the deck's archetype: won, with its standard error (fights)")
    print(f"{'archetype':16} {'':8} " + " ".join(f"{k:>17}" for k in ["all", *used]))
    table = {}
    for a in ORDER:
        rows = archs == a
        if rows.any():
            print(f"{a:16} {'':8} {cell(won[rows])} " + " ".join(cell(won[rows & (kinds == k)]) for k in used))
            table[a] = {k: [int((rows & (kinds == k)).sum()), float(won[rows & (kinds == k)].sum())] for k in used}
    return table


def report_plays(plays, names):
    """(b), returned as JSON too."""
    r = plays.report(names)
    print("\n(b) When it is playable: turns and fights so, the share in which it gets played, plays a fight")
    print(f"{'card':16} {'turns':>7} {'played':>7} {'fights':>7} {'played':>7} {'a fight':>7}")
    for name in names:
        c = r["cards"][name]
        if c["turns"]:
            print(f"{name:16} {c['turns']:7} {c['turns_played'] / c['turns']:7.1%} {c['fights']:7} {c['fights_played'] / max(c['fights'], 1):7.1%} "
                  f"{c['plays'] / max(c['fights'], 1):7.2f}")
        else:
            print(f"{name:16} never playable")
    if "catalyst" in r:
        cat = r["catalyst"]
        plays_on = sum(cat["plays_on"])
        print("\nCatalyst, by poison (the most on an enemy that turn): " + ", ".join(
            f"{p}: played on {k / max(n, 1):.0%} of {n} turns" for p, n, k in zip(cat["poison"], cat["turns"], cat["turns_played"])))
        print(f"Catalyst's {plays_on} plays, by the target's poison: " + ", ".join(f"{p}: {k / max(plays_on, 1):.0%}" for p, k in zip(cat["poison"], cat["plays_on"]))
              + f"; on the most poisoned enemy {cat['on_most_poison'] / max(plays_on, 1):.0%}")
    print("\nCard selections: what gets picked, (in brackets) picking at random among the options")
    for kind, s in sorted(r["selections"].items(), key=lambda x: -x[1]["decisions"]):
        picks = s["decisions"] - s["stops"]
        if picks >= 20:
            print(f"{kind}: {picks} picks, {s['stops']} stops: " + ", ".join(
                f"{c} {s['picked'][c] / picks:.0%} ({s['at_random'][c] / max(s['decisions'], 1):.0%})" for c in CLASSES))
    for name in names:
        rows = r["cards"][name].get("selections", {})
        shown = [f"{k}: picked {v['picked'] / v['offered']:.0%} of {v['offered']} times on offer" for k, v in sorted(rows.items(), key=lambda x: -x[1]["offered"]) if v["offered"] >= 20]
        if shown:
            print(f"{name}: " + "; ".join(shown))
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--scenarios", default=SCENARIOS, help="the scenario lines' files, comma-separated")
    p.add_argument("--cards", default=CARDS, help="names or the game's ids, comma-separated")
    p.add_argument("--kinds", default=None, help="only these fights (boss1,boss2...), comma-separated")
    p.add_argument("--lines", type=int, default=12000, help="fights taken, evenly through the lines (0: all)")
    p.add_argument("--removal-fights", type=int, default=2000, help="(c): fights a card, evenly through those holding it (-1: none, 0: all)")
    p.add_argument("--batch", type=int, default=512, help="fights at once")
    p.add_argument("--gpu-gb", type=float, default=3.5, help="GPU memory cap, GB (PyTorch's own, without the CUDA context's ~0.45 GB; 0 = none)")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="everything as JSON (written after each card)")
    args = p.parse_args()
    device = pick_device(args.device)
    if device.type == "cuda" and args.gpu_gb:
        index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_per_process_memory_fraction(min(args.gpu_gb * 2 ** 30 / torch.cuda.get_device_properties(index).total_memory, 1.0), index)

    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    net, ck = load(args.checkpoint, device)
    net.load_state_dict(ck["net"])
    fight_path = ck["args"].get("fight_net")
    net = load_fight_net(fight_path, device) if fight_path else net.eval()
    character = ck["args"].get("character", "ironclad")
    print(f"{args.checkpoint} (update {ck.get('update')}), {character}, fights by {fight_path or 'itself'}", flush=True)

    names = args.cards.split(",")
    cards = card_ids(names)
    javas = [LAYOUT["card_java_ids"][c] for c in cards]
    fights = read_fights(args.scenarios, character, set(args.kinds.split(",")) if args.kinds else None, args.lines)
    if not fights:
        sys.exit(f"no fights for {character} in {args.scenarios}")
    held = [[i for i, f in enumerate(fights) if any(c[0] == j for c in f["deck"])] for j in javas]

    t0 = time.time()
    plays = Plays(cards, min(args.batch, len(fights)))
    eps = play_fights(net, device, amp, fights, args.batch, character, plays)
    speed = len(fights) / (time.time() - t0)
    print(f"{len(fights)} fights in {time.time() - t0:.0f} s ({speed * 60:.0f} a minute), won {np.mean([e['won'] for e in eps]):.1%}", flush=True)
    result = {"checkpoint": args.checkpoint, "update": ck.get("update"), "fights": len(fights), "wins": report_wins(fights, eps, names, held),
              "archetypes": report_archetypes(fights, eps), "plays": report_plays(plays, names), "removal": {}}

    def write():
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f)

    write()
    if args.removal_fights < 0:
        return
    chosen = [evenly(h, args.removal_fights) for h in held]
    total = sum(len(c) for c in chosen)
    print(f"\n(c) One copy removed, the same fight and seed: {total} fights, about {total / speed / 60:.0f} minutes at that speed. Won with the card, without it, "
          "the difference (with minus without) ± its standard error, in points; `practice_reward` the same; then the difference by kind", flush=True)
    won = np.array([e["won"] for e in eps], dtype=float)
    reward = np.array([practice_reward(e) for e in eps])
    kinds = np.array([kind_of(f) for f in fights])
    groups = {"bosses": np.char.startswith(kinds, "boss"), "elites": np.char.startswith(kinds, "elite"), **{k: kinds == k for k in KINDS[:4]}}
    for name, java, picked in zip(names, javas, chosen):
        t = time.time()
        gone = play_fights(net, device, amp, [without(fights[i], java) for i in picked], args.batch, character)
        if not gone:
            print(f"{name:16} no fights", flush=True)
            continue
        rows = np.array(picked)
        diff = won[rows] - np.array([e["won"] for e in gone], dtype=float)
        gain = reward[rows] - np.array([practice_reward(e) for e in gone])
        by = {}
        for label, group in groups.items():
            mask = group[rows]
            if mask.any():
                m, s = mean_se(diff[mask])
                by[label] = {"fights": int(mask.sum()), "diff": m, "se": s}
        m, s = mean_se(diff)
        rm, rs = mean_se(gain)
        result["removal"][name] = {"fights": len(rows), "with": float(won[rows].mean()), "diff": m, "se": s, "reward": rm, "reward_se": rs, "by": by}
        print(f"{name:16} {len(rows):5} fights: {100 * won[rows].mean():5.1f} {100 * (won[rows] - diff).mean():5.1f}  {100 * m:+5.1f} ±{100 * s:3.1f}  reward {100 * rm:+5.1f} ±{100 * rs:3.1f}  "
              + ", ".join(f"{k} {100 * v['diff']:+.1f}±{100 * v['se']:.1f} ({v['fights']})" for k, v in by.items()) + f"  ({time.time() - t:.0f} s)", flush=True)
        write()


if __name__ == "__main__":
    main()
