"""Checks for routing card picks by the deck's archetype (core.split.ArchetypeRouter, search_eval.py --card-route; plain asserts, no
pytest needed):

    python python/tests/test_card_route.py

What a policy reads from a row: on the card rewards of random runs the deck's archetype is arch's on the game's own view, and the legal
moves are the mask, a Singing Bowl's included. The router: on positions from random runs, every other row and every card reward below its
floor stay the base policy's, plain or a split; at or above it each archetype goes to its picker (decks written into the rows), a deck with
no picker to the default. The max default on made-up logits, the skip and the bowl included. In whole runs: each reward goes where the game's
own view of its deck says, for every archetype (decks given at the start); routes that are all the base network with the base default play
the same runs, one route with the max default a CardPicker's; runs take a defining card under the max default and go to its picker from
then on; sampled picks are drawn the same however the runs are shared between games, only on a picker's rows, from the routed logits. Every
move of whole runs against the game's own view and its real mask, worked out apart from the router: rewards with one to four cards, a
Singing Bowl's, two on one screen. The command lines, end to end, with tiny networks: --free-keys --decks --dump, a split checkpoint, the
look-ahead, two processes, each name's network the checkpoint it names; analysis/archetypes.py --dump on what they wrote; the flags that would
measure nothing or two things refused."""
import collections
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import numpy as np
import torch

from analysis import archetypes
from analysis.archetypes import DISCARD, ORDER, arch
from core.fight_model import MAX_OPTIONS
from core.lookup import CARD_REWARD
from core.run_model import RUN_INT_AT, RunNet
from core.split import CONFIRM, DECK_AT, DECK_END, FLOOR_AT, OPT_KIND_AT, PICK_AT, SCREEN_AT, ArchetypeRouter, CardPicker, SplitPolicy, card_reward_moves, deck_archetypes, fight_rows, max_default
from core.sts_env import ACTIONS, LAYOUT, RUN_INT_LEN, RunEnv
from search import search_eval
from search.search_eval import CardChoices, card_names, parse_card_route, play, route_lines
from tests.test_card_picks import Moves, rows_of, script, tiny_checkpoint

CPU = torch.device("cpu")
NAMES, JAVA_IDS = LAYOUT["card_names"], LAYOUT["card_java_ids"]
PICK_CARD_KIND, BOWL_KIND = (LAYOUT["option_kinds"].index(k) + 1 for k in ("pick_card", "singing_bowl"))
OPT_CARD_AT = RUN_INT_AT["opt_card"][0]
# A deck of each archetype, cards as the game displays them (Envs.set_seed_cards).
DECKS = {"grand finale": ["Grand Finale", "Blade Dance"], "apparitions": ["Apparition", "Acrobatics"], "wraith form": ["Wraith Form"],
         "discard": ["Acrobatics", "Tactician", "Reflex"], "poison": ["Deadly Poison", "Poisoned Stab", "Bouncing Flask"],
         "shiv": ["Blade Dance", "Cloak and Dagger", "Accuracy"], "attacks/mixed": ["Backstab", "Blade Dance"]}


def positions(games=32, steps=400):
    """Rows from random Silent runs: ints, floats, masks, and each row's view (a card reward's; None elsewhere)."""
    env = RunEnv(games, seed=9, ascension=0, character="silent")
    ints, floats, masks, views = [], [], [], []
    for k in range(steps):
        obs = env.observe()
        ints.append(obs["ints"].copy())
        floats.append(obs["floats"].copy())
        masks.append(obs["mask"].copy())
        views += [json.loads(env.envs.observe(g)) if s == CARD_REWARD else None for g, s in enumerate(obs["ints"][:, SCREEN_AT])]
        env.envs.step_random(500 + k)
    return torch.from_numpy(np.concatenate(ints).astype(np.int64)), torch.from_numpy(np.concatenate(floats)), np.concatenate(masks), views


def with_decks(ints, rows, decks):
    """Copies of positions `rows`, the k-th holding the k-th of `decks`, cards as the game displays them."""
    out = ints[rows].clone()
    out[:, DECK_AT:DECK_END] = 0
    for k, deck in enumerate(decks):
        out[k, DECK_AT:DECK_AT + len(deck)] = torch.tensor([NAMES.index(c) + 1 for c in deck])
    return out


def routed_runs(policy, n, batch, sample=None, rule=None, seed=300_000, cards=None, seeded=1.0):
    """`n` runs through `play` (a router writes who decided into the picks), a share `seeded` of them starting with `cards` if given."""
    env = RunEnv(batch, seed=seed, ascension=0, max_steps=1500, character="silent", seeded=seeded if cards else 0.0)
    if cards:
        env.envs.set_seed_cards(cards)
        env.envs.reset()  # (it holds from the next episode, and the first ones are already dealt)
    router = policy if isinstance(policy, ArchetypeRouter) else None
    out, _ = play(policy, CPU, contextlib.nullcontext, env, lambda g, e: e * batch + g < n, None, 0.0, "test", cards=CardChoices(batch, sample=sample, rule=rule, router=router))
    return sorted(out, key=lambda e: e["seed"])


def what_happened(out):
    """The runs without who decided their picks."""
    return [(e["seed"], e["floor"], e["steps"], e["won"], [p[:3] for p in e["picks"]]) for e in out]


class Wants:
    """A stand-in picker: as `Moves` elsewhere, but at a card reward 8 on the slots offering `cards` (as the game displays them), or
    if none of them is offered on the skip."""

    def __init__(self, cards, salt=0):
        self.cards, self.moves = set(cards), Moves(salt)

    def eval(self):
        return self

    def __call__(self, ints, floats, practice=None):
        logits, value = self.moves(ints, floats, practice)
        for row, lg in zip(ints, logits):
            if row[SCREEN_AT] == CARD_REWARD:
                wanted = [PICK_AT + k for k, c in enumerate(row[OPT_CARD_AT:OPT_CARD_AT + MAX_OPTIONS].tolist()) if c and NAMES[c - 1] in self.cards]
                lg[wanted or [CONFIRM]] = 8.0
        return logits, value


def test_read_from_rows():
    # The same labels by a card's id as by its name, an upgrade's + or not: nothing in the rules is spelt for one and not the other.
    for name, java in zip(NAMES, JAVA_IDS):
        assert arch([name] * 3) == arch([java] * 3) and arch([name + "+"]) == arch([java]), name
    assert all(c in JAVA_IDS for c in archetypes.SHIV | archetypes.POISON | archetypes.DISCARD), "a rule's card the rows could never show"
    ints, _, masks, views = positions()
    at_reward = np.flatnonzero(ints[:, SCREEN_AT].numpy() == CARD_REWARD)
    assert len(at_reward) > 50
    labels = deck_archetypes(ints[at_reward])
    for r, label in zip(at_reward, labels):
        assert label == arch(c["name"] for c in views[r]["deck"]), (label, views[r]["deck"])
        assert (ints[r, DECK_AT:DECK_END] > 0).sum() == len(views[r]["deck"]), "the deck field holds the deck as it stands"
    legal, cards = card_reward_moves(ints[at_reward])
    assert np.array_equal(legal.numpy(), masks[at_reward]), "the legal moves as the row says them"
    assert np.array_equal(cards.sum(1).numpy(), [len(views[r]["screen"]["cards"]) for r in at_reward])
    assert not (cards & ~legal).any() and legal[:, CONFIRM].all() and not cards[:, CONFIRM].any()
    # A Singing Bowl's option, in a run's real rows: legal, and not a card. (A boss to beat on floor 16 with the relic in hand.)
    line = json.dumps({"character": "THE_SILENT", "ascension": 0, "act": 1, "floor": 16, "hp": 9000, "max_hp": 9000, "bosses": ["Hexaghost"],
                       "deck": [["Predator", 1]] * 30, "relics": ["Ring of the Snake", "Singing Bowl"], "potions": []})
    env = RunEnv(4, seed=3, ascension=0, character="silent", practice=1.0)
    assert env.envs.set_scenarios([line], 1.0, True, True) == 0
    env.envs.reset()
    bowls = 0
    for k in range(1000):
        obs = env.observe()
        rows = np.flatnonzero(obs["ints"][:, SCREEN_AT] == CARD_REWARD)
        if len(rows):
            row_ints = torch.from_numpy(obs["ints"][rows].astype(np.int64))
            legal, cards = card_reward_moves(row_ints)
            assert np.array_equal(legal.numpy(), obs["mask"][rows])
            bowl = row_ints[:, OPT_KIND_AT:OPT_KIND_AT + MAX_OPTIONS] == BOWL_KIND
            assert legal[:, PICK_AT:PICK_AT + MAX_OPTIONS][bowl].all() and not cards[:, PICK_AT:PICK_AT + MAX_OPTIONS][bowl].any()
            bowls += int(bowl.sum())
        env.envs.step_random(500 + k)
    assert bowls > 3
    print(f"from the rows: {len(at_reward)} card rewards' archetypes and legal moves as the game's view has them, {collections.Counter(labels).most_common(2)}; {bowls} with a Singing Bowl")


def test_router_rows():
    ints, floats, _, _ = positions()
    practice = torch.zeros(len(ints), dtype=torch.bool)
    tiny = lambda: RunNet(d=16, layers=1, heads=2).eval()  # noqa: E731
    base = tiny()
    pickers = {a: tiny() for a in ORDER}
    at_reward = ints[:, SCREEN_AT] == CARD_REWARD
    floor = 1 + int(floats[at_reward, FLOOR_AT].median())
    late = at_reward & (floats[:, FLOOR_AT] >= floor)
    assert 10 < late.sum() < at_reward.sum() and fight_rows(ints).any()
    with torch.no_grad():
        lb, vb = base(ints, floats, practice)
        # Every other row and every card reward below the floor: the base policy's, whichever default, whichever pickers. (Neow's
        # rewards too, on floor 0, with the lowest floor.)
        for default in ("base", "max"):
            for names, low in ((ORDER, floor), (["poison", "discard"], floor), (ORDER, 1)):
                router = ArchetypeRouter(base, {a: pickers[a] for a in names}, low, default)
                lr, vr = router(ints, floats, practice)
                rows = at_reward & (floats[:, FLOOR_AT] >= low)
                assert torch.equal(lr[~rows], lb[~rows]) and torch.equal(vr[~rows], vb[~rows]), "the fights, other choices, the floors below"
                assert not router.rows(ints, floats)[~rows].any()
        # At or above it, by the deck (written into real rows of rewards): each archetype to its own picker, all in one batch.
        rows = torch.nonzero(late).squeeze(1)
        which = [ORDER[k % len(ORDER)] for k in range(len(rows))]
        di, df, dp = with_decks(ints, rows, [DECKS[w] for w in which]), floats[rows], practice[rows]
        assert deck_archetypes(di) == which, "the decks we meant"
        lr, vr = ArchetypeRouter(base, pickers, floor, "base")(di, df, dp)
        for name, net in pickers.items():
            part = torch.tensor([k for k, w in enumerate(which) if w == name])
            lp, vp = net(di[part], df[part], dp[part])
            assert torch.equal(lr[part], lp) and torch.equal(vr[part], vp), name
        # A deck with no picker: to the default. "base": the base policy's. "max": the pickers together, with the base policy's value.
        named = {a: pickers[a] for a in ("poison", "discard")}
        lab, vab = base(di, df, dp)
        rest = torch.tensor([k for k, w in enumerate(which) if w not in named])
        for default in ("base", "max"):
            router = ArchetypeRouter(base, named, floor, default)
            lr, vr = router(di, df, dp)
            who = router.deciders(di, df)
            for name, net in named.items():
                part = torch.tensor([k for k, w in enumerate(which) if w == name])
                assert torch.equal(lr[part], net(di[part], df[part], dp[part])[0]) and all(who[k] == name for k in part.tolist()), name
            assert torch.equal(vr[rest], vab[rest])
            assert torch.equal(router.rows(di, df), torch.tensor([k in who for k in range(len(di))]))
            if default == "base":
                assert torch.equal(lr[rest], lab[rest]) and not set(rest.tolist()) & set(who)
            else:
                together = max_default(torch.stack([net(di[rest], df[rest], dp[rest])[0] for net in named.values()]), *card_reward_moves(di[rest]))
                assert torch.equal(lr[rest], together) and all(who[k] == "max" for k in rest.tolist())
        # On a split, the fights are the fight network's, the other choices the deck network's, the rewards the picker's.
        fight = tiny()
        combat = fight_rows(ints)
        split = SplitPolicy(fight, base)
        ls, _ = ArchetypeRouter(split, pickers, floor, "max")(ints, floats, practice)
        lf, _ = split(ints, floats, practice)
        assert torch.equal(ls[~late], lf[~late]) and torch.equal(ls[combat], fight(ints[combat], floats[combat], practice[combat])[0])
        assert not torch.equal(ls[late], lf[late])
        # Under autocast, as `play` runs them on the GPU: the router's logits are floats, its picks the same but for rounding.
        router = ArchetypeRouter(base, named, floor, "max")
        plain = router(di, df, dp)[0]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            rounded = router(di, df, dp)[0]
        assert rounded.dtype == torch.float32 and (rounded.argmax(-1) == plain.argmax(-1)).float().mean() > 0.8
    # For evals: eval reaches its networks, and it has no training mode.
    base.train()
    pickers["poison"].train()
    router = ArchetypeRouter(base, {"poison": pickers["poison"]}).eval()
    assert not base.training and not pickers["poison"].training
    try:
        router.train()
    except RuntimeError:
        pass
    else:
        raise AssertionError("train() ran")
    print(f"the router: {int(late.sum())} card rewards from floor {floor} on to the deck's picker or the default; the other {int((~late).sum())} rows the base policy's")


def test_max_default():
    legal = torch.zeros(1, ACTIONS, dtype=torch.bool)
    legal[0, PICK_AT:PICK_AT + 3] = True
    legal[0, CONFIRM] = True
    cards = legal.clone()
    cards[0, CONFIRM] = False

    def probs(*pickers, legal=legal):
        """The max default's probabilities for one row's legal moves in slot order, from each picker's: its logits, with big ones on the
        illegal slots, different for each picker (a softmax over them too would scale each picker differently)."""
        rows = []
        for k, q in enumerate(pickers):
            lg = torch.full((1, ACTIONS), 40.0 - 20.0 * k)
            lg[legal] = torch.tensor(q).log()
            rows.append(lg)
        out = max_default(torch.stack(rows), legal, cards)
        assert (out[~legal] == -1e9).all(), "only legal moves"
        return out.exp()[legal].tolist()

    def close(a, b):
        assert np.allclose(a, b, atol=1e-5), (a, b)

    # Each card its largest, the skip its smallest (these happen to add up to one).
    close(probs([0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]), [0.7, 0.1, 0.1, 0.1])
    # Renormalised.
    close(probs([0.4, 0.1, 0.1, 0.4], [0.1, 0.4, 0.1, 0.4]), [0.4 / 1.3, 0.4 / 1.3, 0.1 / 1.3, 0.4 / 1.3])
    # One picker wanting a card takes it from the two that want to skip.
    p = probs([0.05, 0.05, 0.05, 0.85], [0.05, 0.05, 0.05, 0.85], [0.05, 0.5, 0.05, 0.4])
    close(p, [0.05, 0.5, 0.05, 0.4])
    assert int(np.argmax(p)) == 1, p
    # The skip if every picker wants to.
    p = probs([0.1, 0.1, 0.1, 0.7], [0.05, 0.05, 0.1, 0.8], [0.15, 0.1, 0.05, 0.7])
    close(p, [0.15 / 1.05, 0.1 / 1.05, 0.1 / 1.05, 0.7 / 1.05])
    assert int(np.argmax(p)) == 3, p
    # One picker: its own masked softmax.
    close(probs([0.2, 0.3, 0.1, 0.4]), [0.2, 0.3, 0.1, 0.4])
    # A Singing Bowl's +2 Max HP is no card: every picker has to want it, as with the skip.
    bowl = legal.clone()
    bowl[0, PICK_AT + 3] = True
    close(probs([0.1, 0.1, 0.1, 0.6, 0.1], [0.1, 0.1, 0.1, 0.1, 0.6], legal=bowl), [0.2] * 5)
    # Row by row in a batch; a probability of about nothing stays finite.
    both = legal.repeat(2, 1)
    a, b = torch.full((2, ACTIONS), 40.0), torch.full((2, ACTIONS), 40.0)
    a[both] = torch.tensor([9.0, 0, 0, 0, -300.0, -300.0, -300.0, 0])
    b[both] = torch.tensor([0.0, 9.0, 0, 0, -300.0, -300.0, -300.0, 0])
    out = max_default(torch.stack([a, b]), both, cards.repeat(2, 1))
    assert out[0].argmax() in (PICK_AT, PICK_AT + 1) and out[1].argmax() == CONFIRM and torch.isfinite(out).all(), out[both]
    # The row's own options, read as card_reward_moves reads them.
    row = torch.zeros(1, RUN_INT_LEN, dtype=torch.long)
    row[0, OPT_KIND_AT:OPT_KIND_AT + 4] = torch.tensor([PICK_CARD_KIND, PICK_CARD_KIND, PICK_CARD_KIND, BOWL_KIND])
    got_legal, got_cards = card_reward_moves(row)
    assert torch.equal(got_legal, bowl) and torch.equal(got_cards, cards)
    print("the max default: each card its largest, a skip or a bowl its smallest, renormalised, on the legal moves alone")


def test_runs_go_where_the_view_says():
    """Whole runs with a deck given at the start: every reward's "how" is what arch makes of the game's own view of its deck then."""
    pickers = {a: Moves(k + 1) for k, a in enumerate(ORDER[:-1])}  # every archetype's but attacks or mixed
    seen = collections.Counter()
    for name, deck in DECKS.items():
        for default, floor in (("max", 1), ("base", 3)):
            router = ArchetypeRouter(Moves(), pickers, floor, default)
            rewards = []

            def note(view, logits):
                rewards.append((view["floor"], card_names(view["screen"]["cards"]), arch(c["name"] for c in view["deck"])))

            rows = rows_of(routed_runs(router, 12, 6, rule=note, cards=deck))
            assert rows and rewards, name
            wanted = collections.Counter((f, tuple(c), "own" if f < floor else a if a in pickers else default if default == "max" else "own") for f, c, a in rewards)
            assert wanted == collections.Counter((r[0], tuple(r[1]), r[3]) for r in rows), name
            seen.update(r[3] for r in rows)
            assert any(r[0] >= floor and r[3] == (name if name in pickers else default if default == "max" else "own") for r in rows), name
    assert all(seen[a] for a in pickers) and seen["max"] and seen["own"], seen
    print(f"whole runs with a deck of each archetype: every reward's picker as the game's view of its deck says: {dict(seen)}")


class Checked(RunEnv):
    """A RunEnv that checks each move played on it, worked out from the game's own view and the real mask, with no deck_archetypes,
    card_reward_moves or max_default: every other row the base policy's; a card reward its decider's, each network asked on that row alone;
    the card taken is what joins the deck, and a skip or a bowl adds none. `expected`: what each run's picks should say, by its seed."""

    def watch(self, base, pickers, floor, default, wanted, seed):
        self.base, self.pickers, self.floor, self.default, self.wanted, self.seed = base, pickers, floor, default, wanted, seed
        self.episode = np.zeros(self.n, dtype=int)
        self.expected = collections.defaultdict(list)
        self.counts = collections.Counter()
        return self

    def observe(self):
        self.last = super().observe()
        return self.last

    def decider(self, view):
        label = arch(c["name"] for c in view["deck"])
        if view["floor"] < self.floor:
            return "own"
        return label if label in self.pickers else "max" if self.default == "max" else "own"

    def by_hand(self, who, row, n_cards, legal):
        """The logits `who` should give for one row: (ints, floats, practice), the row's real legal moves."""
        if who == "own":
            return self.base(*row)[0][0]
        if who != "max":
            return self.pickers[who](*row)[0][0]
        probs = []
        for net in self.pickers.values():
            p = torch.zeros(ACTIONS)
            p[legal] = torch.softmax(net(*row)[0][0][legal], -1)
            probs.append(p)
        probs = torch.stack(probs)
        q = torch.zeros(ACTIONS)
        for a in legal.tolist():
            q[a] = probs[:, a].max() if PICK_AT <= a < PICK_AT + n_cards else probs[:, a].min()
        return (q / q.sum()).log().clamp_min(-1e9)

    def step(self, acts):
        ints, floats = torch.from_numpy(self.last["ints"]).long(), torch.from_numpy(self.last["floats"])
        practice = torch.zeros(self.n, dtype=torch.bool)
        base = self.base(ints, floats, practice)[0].masked_fill(~torch.from_numpy(self.last["mask"]), -1e9).argmax(-1).numpy()
        decks = {}
        for g in range(self.n):
            run = self.seed + self.episode[g] * self.n + g
            if not self.wanted(g, self.episode[g]):
                continue
            if self.last["ints"][g, SCREEN_AT] != CARD_REWARD:
                assert acts[g] == base[g], ("not a card reward: the base policy's move", g, acts[g], base[g])
                self.counts["other rows"] += 1
                continue
            view = json.loads(self.envs.observe(int(g)))
            offered = card_names(view["screen"]["cards"])
            n, bowl = len(offered), "Singing Bowl" in view["relics"]
            legal = np.flatnonzero(self.last["mask"][g])
            assert legal.tolist() == [PICK_AT + k for k in range(n + bowl)] + [CONFIRM], (legal, n, bowl)
            who = self.decider(view)
            logits = self.by_hand(who, (ints[[g]], floats[[g]], practice[[g]]), n, torch.from_numpy(legal))
            assert acts[g] == legal[logits[legal].argmax()], (who, view["floor"], offered, acts[g], logits[legal])
            taken = offered[acts[g] - PICK_AT] if acts[g] < PICK_AT + n else None
            self.expected[run].append([view["floor"], offered, taken, who])
            self.counts[who] += 1
            self.counts[f"{n} cards"] += 1
            self.counts["with a bowl"] += bowl
            decks[g] = (collections.Counter(c["name"].rstrip("+") for c in view["deck"]), taken)
        status = super().step(acts)
        for g, (deck, taken) in decks.items():
            if not status[g]:
                after = collections.Counter(c["name"].rstrip("+") for c in json.loads(self.envs.observe(int(g)))["deck"])
                assert after - deck == collections.Counter([taken] if taken else []) and not deck - after, (taken, after - deck, deck - after)
        self.episode += status != 0
        return status


def checked_runs(pickers, floor, default, n=16, batch=8, seed=300_000, cards=None, line=None):
    """`n` runs through `play` on a Checked env, a router's, starting with `cards` or from a boss `line`; returns (the runs, the env)."""
    env = Checked(batch, seed=seed, ascension=0, max_steps=1500, character="silent", seeded=1.0 if cards else 0.0, practice=1.0 if line else 0.0)
    if cards:
        env.envs.set_seed_cards(cards)
    if line:
        assert env.envs.set_scenarios([line], 1.0, True, True) == 0
    env.envs.reset()
    base = Moves()
    wanted = lambda g, e: e * batch + g < n  # noqa: E731
    env.watch(base, pickers, floor, default, wanted, seed)
    router = ArchetypeRouter(base, pickers, floor, default)
    out, _ = play(router, CPU, contextlib.nullcontext, env, wanted, None, 0.0, "test", cards=CardChoices(batch, router=router))
    assert len(out) == n
    for e in out:
        assert [r for r in e["picks"] if r[3] != "unopened"] == env.expected[e["seed"]], e["seed"]
    return out, env


def test_every_move():
    pickers = {"poison": Moves(1), "discard": Wants([n for n, j in zip(NAMES, JAVA_IDS) if j in DISCARD], 2)}
    seen = collections.Counter()
    for default, floor, cards in (("max", 3, None), ("base", 3, None), ("max", 1, ["Acrobatics+", "Tactician"]), ("max", 1, ["Deadly Poison+", "Crippling Cloud", "Poisoned Stab+"]),
                                  ("base", 2, ["Sneaky Strike+", "Tools of the Trade", "Unload"])):
        _, env = checked_runs(pickers, floor, default, cards=cards)
        seen.update(env.counts)
    assert all(seen[k] for k in ("other rows", "own", "max", "poison", "discard")), seen

    # Rewards of other sizes on real masks, from a boss with the relic in hand: 4 cards and a bowl, 1 card and a bowl, 1 card alone.
    def line(*relics, deck=()):
        return json.dumps({"character": "THE_SILENT", "ascension": 0, "act": 1, "floor": 16, "hp": 9000, "max_hp": 9000, "bosses": ["Hexaghost"],
                           "deck": [["Predator", 1]] * 30 + list(deck), "relics": ["Ring of the Snake", *relics], "potions": []})

    for relics in (("Question Card", "Singing Bowl"), ("Busted Crown", "Singing Bowl"), ("Busted Crown",)):
        _, env = checked_runs(pickers, 1, "max", n=4, batch=4, line=line(*relics))
        seen.update(env.counts)
    assert all(seen[k] for k in ("1 cards", "2 cards", "3 cards", "4 cards", "with a bowl")), seen
    # Two rewards on a screen: one taken, and the next, on the same floor, already goes by the deck that holds it.
    out, env = checked_runs(pickers, 1, "max", n=8, batch=4, line=line("Prayer Wheel", deck=[["Acrobatics", 0], ["Tactician", 1]]))
    rows = [[r for r in e["picks"] if r[3] != "unopened"] for e in out]
    flips = [(a, b) for run in rows for a, b in zip(run, run[1:]) if a[0] == b[0] and a[3] == "max" and b[3] == "discard"]
    assert flips and all(archetypes.JAVA_ID.get(a[2], a[2]) in DISCARD for a, _ in flips), flips
    print(f"every move of whole runs by hand, on real masks: {dict(seen)}; {len(flips)} rewards went to the discard picker straight after the card that made the deck one, on the same floor")


def test_same_runs():
    base = Moves()
    alone = routed_runs(base, 24, 12)
    assert len(rows_of(alone)) > 10
    # All the routes the base network, the default the base: the same runs. So with a floor no run reaches.
    same = routed_runs(ArchetypeRouter(base, {a: base for a in ORDER}, 1, "base"), 24, 12)
    assert what_happened(same) == what_happened(alone)
    assert {r[3] for r in rows_of(same)} <= {*ORDER, "own"}, "the base network's, written down as who decided"
    assert routed_runs(ArchetypeRouter(base, {"poison": Moves(1), "discard": Moves(2)}, 1000, "max"), 24, 12) == alone, "the same summaries, the picks' \"own\" too"
    # One route and the max default: that picker's picks, as a CardPicker's, its draws too.
    other = Moves(1)
    picker, route = CardPicker(base, other), ArchetypeRouter(base, {"poison": other}, 1, "max")
    assert what_happened(routed_runs(route, 24, 12)) == what_happened(routed_runs(picker, 24, 12))
    draws = lambda p: (p, 1, lambda g, e: 300_000 + e * 12 + g)  # noqa: E731
    assert what_happened(routed_runs(route, 24, 12, draws(route))) == what_happened(routed_runs(picker, 24, 12, draws(picker)))
    print("the same runs with routes that are all the base network and the base default; one route plays as a CardPicker")


def test_commit():
    """Under the max default a run takes what a picker wants, and with that its deck's archetype is that picker's."""
    wants = [n for n, j in zip(NAMES, JAVA_IDS) if j in DISCARD]
    pickers = {"discard": Wants(wants), "grand finale": Wants(["Grand Finale"], 1)}
    router = ArchetypeRouter(Moves(screens={CARD_REWARD: CONFIRM}), pickers, 1, "max")
    took = committed = 0
    for e in routed_runs(router, 96, 16, cards=["Acrobatics", "Tactician"]):
        rows = [r for r in e["picks"] if r[3] not in ("unopened", "own")]
        for r, after in zip(rows, rows[1:] + [None]):
            wanted = [c for c in r[1] if c in (pickers[r[3]].cards if r[3] in pickers else {*wants, "Grand Finale"})]
            assert r[2] in wanted if wanted else r[2] is None, r
            if r[3] == "max" and r[2] in wants and after:
                took += 1
                committed += after[3] == "discard"
    # (Not always: a run's random Neow or event can have taken one of the two given.)
    assert took >= 5 and committed >= 0.7 * took, (took, committed)
    print(f"runs commit: {committed} of {took} that took a third discard card under the max default went on with the discard picker")


def test_sampling():
    def leaning(skip):
        """A stand-in picker: every card alike, the skip `skip` times as likely as a card."""
        def net(ints, floats, practice=None):
            lg = torch.zeros(len(ints), ACTIONS)
            lg[:, CONFIRM] = np.log(skip)
            return lg, torch.zeros(len(ints))

        net.eval = lambda: net
        return net

    base = Moves(screens={CARD_REWARD: PICK_AT})
    flat, shy = leaning(1.0), leaning(3.0)
    router = ArchetypeRouter(base, {"poison": flat, "discard": flat}, 1, "max")
    sample = lambda r, s, b: (r, s, lambda g, e: 300_000 + e * b + g)  # noqa: E731
    one = routed_runs(router, 32, 8, sample(router, 1, 8))
    rows = rows_of(one)
    slots = [r[1].index(r[2]) if r[2] else -1 for r in rows]
    counts = [slots.count(k) for k in (-1, 0, 1, 2)]
    assert len(rows) > 30 and min(counts) > 0.1 * len(rows), counts
    # Reproducible, and the same however the runs are shared out; another seed, other draws.
    assert one == routed_runs(router, 32, 8, sample(router, 1, 8))
    assert [e["picks"] for e in routed_runs(router, 32, 4, sample(router, 1, 4))] == [e["picks"] for e in one], "by run, not by game"
    assert [e["picks"] for e in routed_runs(router, 32, 8, sample(router, 2, 8))] != [e["picks"] for e in one]

    # The draw follows the routed logits: among three cards and a skip, half skips with both pickers shy of cards, a quarter with only one.
    def skips(*nets):
        r = ArchetypeRouter(base, dict(zip(("poison", "discard"), nets)), 1, "max")
        drawn = [x for x in rows_of(routed_runs(r, 96, 8, sample(r, 1, 8))) if len(x[1]) == 3 and x[3] == "max"]
        assert len(drawn) > 80, len(drawn)
        return sum(x[2] is None for x in drawn) / len(drawn)

    both, one_shy = skips(shy, shy), skips(shy, flat)
    assert 0.4 < both < 0.6 and 0.15 < one_shy < 0.35, (both, one_shy)
    # Only a picker's rows draw: the rest keep the base policy's top pick, its first card here.
    router = ArchetypeRouter(base, {"poison": flat}, 1, "base")
    mixed = rows_of(routed_runs(router, 64, 8, sample(router, 1, 8), cards=DECKS["poison"], seeded=0.5))
    drawn, kept = [r for r in mixed if r[3] == "poison"], [r for r in mixed if r[3] == "own"]
    assert len(drawn) > 10 and len(kept) > 10 and all(r[2] == r[1][0] for r in kept), (len(drawn), len(kept))
    assert any(r[2] != r[1][0] for r in drawn)
    # Below the floor too.
    router = ArchetypeRouter(base, {"poison": flat, "discard": flat}, 3, "max")
    rows = rows_of(routed_runs(router, 32, 8, sample(router, 1, 8)))
    low = [r for r in rows if r[0] < 3]
    assert low and all(r[3] == "own" and r[2] == r[1][0] for r in low) and any(r[0] >= 3 and r[2] != r[1][0] for r in rows)
    print(f"--card-sample: {counts} skips and first, second, third cards from the routed logits; the same draws at 4 and 8 games; skips {both:.0%} and {one_shy:.0%} as the pickers lean")


def test_tables():
    out = [{"picks": [[3, ["Slice", "Bane"], "Slice", "own"], [20, ["Bane", "Dash"], "Dash", "max"], [21, ["Bane"], None, "max"], [36, ["Bane"], "Bane", "poison"], [36, ["Dash"], None, "unopened"]]}]
    lines = route_lines(out, ["poison", "discard", "max", "own"])
    assert lines[0] == "  decided by poison: 1 of 4 card rewards, Act 3 1; skipped 0%" and lines[1] == "    Act 3: skipped 0%; most taken: Bane 100% of 1", lines
    assert lines[2] == "  decided by discard: 0 of 4 card rewards; skipped 0%", lines
    assert lines[3] == "  decided by max: 2 of 4 card rewards, Act 2 2; skipped 50%" and lines[4] == "    Act 2: skipped 50%; most taken: Dash 100% of 1", lines
    assert lines[5] == "  decided by own: 1 of 4 card rewards, floors 1-8 1; skipped 0%", lines
    names = {"grand-finale": "grand finale", "apparitions": "apparitions", "wraith-form": "wraith form", "discard": "discard", "poison": "poison", "shiv": "shiv", "attacks/mixed": "attacks/mixed"}
    assert parse_card_route(",".join(f"{k}=x/{k}.pt" for k in names)) == {v: f"x/{k}.pt" for k, v in names.items()}
    assert parse_card_route("poison=~/a=b.pt") == {"poison": os.path.expanduser("~/a=b.pt")}
    assert lines[7] == "  left unopened by the checkpoint, no picker's: 1, Act 3 1" and len(lines) == 8, lines
    assert route_lines([{"picks": out[0]["picks"][:1]}], ["poison", "own"])[-1].startswith("    floors 1-8"), "no line when none were left"
    for bad in ("", "nope=x.pt", "grand finale=x.pt", "poison", "poison=", "poison=x.pt,poison=y.pt", "poison=x.pt,"):
        try:
            parse_card_route(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} parsed")


def test_scripts():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        plain, fight, split, lookup, card_a, card_b = (os.path.join(d, f"{n}.pt") for n in ("plain", "fight", "split", "lookup", "a", "b"))
        for path in (plain, fight, card_a, card_b):
            tiny_checkpoint(path)
        tiny_checkpoint(split, fight_net=fight, lookup=False)
        tiny_checkpoint(lookup, lookup=True)
        small = ["--character", "silent", "--set", "runs", "--no-search", "--ascension", "0", "--device", "cpu", "--seed-offset", "1024"]
        route = f"grand-finale={card_a},wraith-form={card_b},discard={card_a},poison={card_b}"
        prefix = os.path.join(d, "dump")

        def dumped():
            return sorted((json.loads(line) for line in open(prefix + ".0.jsonl")), key=lambda e: e["seed"])

        # Every flag at once on a plain checkpoint: the keys, the decks, the dump, the counts, sampled picks.
        out = script(search_eval.main, plain, *small, "--runs", "24", "--batch", "12", "--free-keys", "--decks", "--card-route", route, "--card-from-floor", "2", "--card-sample", "--dump", prefix)
        assert "card picks by the deck's archetype from floor 2: grand finale " in out and "any other deck: max; sampled" in out and f"poison {card_b} (step 1)" in out, out
        assert "floors 1-8" in out and "runs A0 greedy: won" in out and "its bosses beaten" in out, out
        for who in ("grand finale", "wraith form", "discard", "poison", "max", "own"):
            assert f"decided by {who}: " in out, out
        runs = dumped()
        assert len(runs) == 24 and all("picks" in e and "deck" in e and e["keys"] >= 3 for e in runs)
        hows = {r[3] for e in runs for r in e["picks"]}
        assert hows <= {"grand finale", "wraith form", "discard", "poison", "max", "own", "unopened"} and "max" in hows, hows
        assert all(r[3] in ("own", "unopened") for e in runs for r in e["picks"] if r[0] < 2), "below the floor"
        # analysis/archetypes.py --dump on that: each archetype's share of the runs, its Act 3, pair and Heart rates (--min-floor 1: every run).
        table = script(archetypes.main, "--dump", prefix, "--min-floor", "1")
        assert f"{prefix}: 24 runs, 24 past floor 0 with a deck" in table and "Act 3" in table and "the pair" in table and "the Heart" in table, table
        assert abs(sum(float(line.split("%")[0].split()[-1]) for line in table.splitlines() if line.startswith("  ")) - 100) < 0.5, table
        for a in {arch(e["deck"]) for e in runs}:
            sizes = [len(e["deck"]) for e in runs if arch(e["deck"]) == a]
            assert any(line.strip().startswith(a) and f" {len(sizes):5} runs " in line and line.endswith(f"deck {sum(sizes) / len(sizes):4.1f}") for line in table.splitlines()), (a, table)

        # A checkpoint with itself as every route and the base default: the same runs as with no route at all, a real network's.
        common = [*small, "--runs", "6", "--batch", "3", "--picks", "--dump", prefix]
        script(search_eval.main, plain, *common)
        no_route = what_happened(dumped())
        out = script(search_eval.main, plain, *common, "--card-route", ",".join(f"{a.replace(' ', '-')}={plain}" for a in ORDER), "--card-route-default", "base")
        assert what_happened(dumped()) == no_route and "any other deck: base; top pick" in out and "decided by max" not in out, out
        # Other pickers change them, and their draws don't care how the runs are shared between games.
        script(search_eval.main, plain, *common, "--card-route", route)
        one = what_happened(dumped())
        assert one != no_route
        draws = [*small, "--runs", "6", "--dump", prefix, "--card-route", route, "--card-sample"]
        script(search_eval.main, plain, *draws, "--batch", "3")
        three = [e["picks"] for e in dumped()]
        script(search_eval.main, plain, *draws, "--batch", "6")
        assert three == [e["picks"] for e in dumped()]
        # The base default reaches the router: rewards opened, none to the pickers together.
        script(search_eval.main, plain, *small, "--runs", "24", "--batch", "12", "--dump", prefix, "--card-route", route, "--card-route-default", "base")
        rows = rows_of(dumped())
        assert sum(r[3] == "own" for r in rows) > 10 and "max" not in {r[3] for r in rows}, rows[:5]

        # A split checkpoint, its fighter and its look-ahead: the fights keep their network, whatever picks the cards.
        out = script(search_eval.main, split, *small, "--runs", "3", "--batch", "3", "--fighter", fight, "--card-route", route, "--card-route-default", "base")
        assert "decided by own" in out and "decided by max" not in out, out
        look_ahead = [arg for arg in small if arg != "--no-search"]
        out = script(search_eval.main, split, *look_ahead, "--no-baseline", "--no-fixed", "--runs", "2", "--batch", "2", "--k", "1", "--top", "1", "--max-moves", "3", "--card-route", route, "--card-from-floor", "2")
        assert "runs A0 search: won" in out and "decided by poison" in out, out
        # Two processes: the same runs as one, the draws too.
        def two_processes(*flags):
            script(search_eval.main, plain, *common, "--procs", "2", *flags)
            return sorted((json.loads(line) for i in range(2) for line in open(f"{prefix}.{i}.jsonl")), key=lambda e: e["seed"])

        assert what_happened(two_processes("--card-route", route)) == one
        assert [e["picks"] for e in two_processes("--card-route", route, "--card-sample", "--batch", "6")] == three

        # Each name's network is the checkpoint it names.
        named = {}

        class Spy(ArchetypeRouter):
            def __init__(self, policy, pickers, *rest):
                named.update(pickers)
                super().__init__(policy, pickers, *rest)

        search_eval.ArchetypeRouter = Spy
        try:
            script(search_eval.main, plain, *small, "--runs", "1", "--batch", "1", "--card-route", route)
        finally:
            search_eval.ArchetypeRouter = ArchetypeRouter
        assert len(named) == 4
        for a, path in (("grand finale", card_a), ("wraith form", card_b), ("discard", card_a), ("poison", card_b)):
            weights = torch.load(path, map_location="cpu", weights_only=False)["net"]
            assert all(torch.equal(v, weights[k]) for k, v in named[a].state_dict().items()), a
        assert named["discard"] is named["grand finale"], "one network for a checkpoint named twice"

        # Refused: two things choosing at card rewards, a name that isn't an archetype's, a lookup nothing fills, flags that would do nothing.
        args = [plain, *small, "--runs", "3", "--batch", "3"]
        for bad in ([*args, "--card-route", f"grand-finale={card_a}", "--card-net", card_b],
                    [*args, "--card-route", f"grand-finale={card_a}", "--peek-margin", "0.1"],
                    [*args, "--card-route", f"grand finale={card_a}"],
                    [*args, "--card-route", f"wraith={card_a}"],
                    [*args, "--card-route", f"poison={lookup}"],
                    [*args, "--card-route", f"poison={card_a},discard={lookup}"],
                    [*args, "--card-route", f"poison={card_a},poison={card_b}"],
                    [*args, "--card-route", ""],
                    [*args, "--card-route-default", "base"],
                    [*args, "--card-from-floor", "17"],
                    [*args, "--card-route", f"poison={card_a}", "--set", "bosses"],
                    [plain, "--set", "runs", "--no-search", "--runs", "3", "--device", "cpu", "--card-route", f"poison={card_a}"]):  # the Ironclad, by default
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    script(search_eval.main, *bad)
            except SystemExit as e:
                assert e.code == 2, bad
                continue
            raise AssertionError(f"{bad} ran")
    print("the command lines: search_eval --card-route with --free-keys --decks --dump --card-sample --procs, a split, the look-ahead; archetypes.py --dump")


if __name__ == "__main__":
    test_read_from_rows()
    test_router_rows()
    test_max_default()
    test_runs_go_where_the_view_says()
    test_every_move()
    test_same_runs()
    test_commit()
    test_sampling()
    test_tables()
    test_scripts()
    print("ok")
