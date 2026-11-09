"""The split (notes/split-networks-plan.md, D1-D2): one network plays the fights, another makes the
choices between fights.

    policy = SplitPolicy(fight, deck)        # each a RunNet, or the FixedShapes of one
    logits, value = policy(ints, floats, practice)

Fight rows (the combat screen, so practice fights too) take the fight network's logits and value,
every other row the deck network's. With both FixedShapes the batch goes through the fixed passes
once, each kind through its own network's (`FixedShapes.__call__(..., deck=)`): no row runs twice.
Otherwise each network runs on its own rows only.
`train()` reaches the network that trains only (`trains`: "deck", the default, train_runs.py --fight-net; or "fight",
train_runs.py --deck-net): the other stays in eval mode (it is held fixed).

`Ensemble(nets)` plays as their mean logits (and mean value): several deck networks' checkpoints as one deck network
(search/search_eval.py: a comma-separated checkpoint list). `BossFighter(policy, boss)` gives the boss fights' rows to another
fight network (search_eval.py --boss-fighter). `CardPicker(policy, cards, from_floor)` gives the card reward rows to another
network, from a floor on (search_eval.py --card-net). `ArchetypeRouter(policy, pickers, from_floor, default)` gives each one to the
picker for the archetype of that run's deck as it stands (search_eval.py --card-route).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import torch

from analysis.archetypes import arch
from core.fixed_shapes import FixedShapes
from core.lookup import CARD_REWARD
from core.run_model import COMBAT, RUN_FLOAT_AT, RUN_INT_AT
from core.sts_env import ACTIONS, LAYOUT

SCREEN_AT = RUN_INT_AT["screen"][0]
ROOM_AT = RUN_INT_AT["room"][0]
BOSS_ROOM = 7  # encode_run.rs room_id + 1
FLOOR_AT = RUN_FLOAT_AT["run_scalars"][0] + 3  # hp, max hp, gold, floor, act
ACT_AT = RUN_FLOAT_AT["run_scalars"][0] + 4
DECK_AT, DECK_END = RUN_INT_AT["deck"]
OPT_KIND_AT, OPT_KIND_END = RUN_INT_AT["opt_kind"]
PICK_CARD = LAYOUT["option_kinds"].index("pick_card") + 1  # encode_run.rs kind
PICK_AT, CONFIRM = LAYOUT["actions"]["pick"][0], LAYOUT["actions"]["confirm"]


def fight_rows(ints):
    """Which rows are fight rows (the combat screen)."""
    return ints[:, SCREEN_AT] == COMBAT


class SplitPolicy:
    def __init__(self, fight, deck, trains="deck"):
        assert trains in ("deck", "fight")
        self.fight, self.deck, self.trains = fight, deck, trains

    def eval(self):
        self.fight.eval()
        self.deck.eval()
        return self

    def train(self, mode=True):
        (self.deck if self.trains == "deck" else self.fight).train(mode)
        return self

    def __call__(self, ints, floats, practice=None):
        if isinstance(self.fight, FixedShapes) and isinstance(self.deck, FixedShapes):
            return self.fight(ints, floats, practice, deck=self.deck)
        # Each network sees only its own rows (the eval worker plays on the CPU: both on every row
        # took twice as long, 2026-10-29).
        combat = fight_rows(ints)
        parts = [(net, torch.nonzero(rows).squeeze(1)) for net, rows in ((self.fight, combat), (self.deck, ~combat))]
        logits = value = None
        for net, rows in parts:
            if len(rows) == 0:
                continue
            lg, v = net(ints[rows], floats[rows], practice[rows] if practice is not None else None)
            if logits is None:
                logits = lg.new_empty((len(ints), lg.shape[1]))
                value = v.new_empty(len(ints))
            logits[rows], value[rows] = lg.to(logits.dtype), v.to(value.dtype)
        return logits, value


class Ensemble:
    """Several networks as one: their logits and values averaged (logits: a product of the networks' move distributions,
    up to each one's normalisation; for greedy play and sampling alike). For evals (no training)."""

    def __init__(self, nets):
        self.nets = list(nets)

    def eval(self):
        for n in self.nets:
            n.eval()
        return self

    def train(self, mode=True):
        raise RuntimeError("Ensemble is for evals")

    def __call__(self, ints, floats, practice=None):
        outs = [n(ints, floats, practice) for n in self.nets]
        logits = torch.stack([o[0].float() for o in outs]).mean(0)
        value = torch.stack([o[1].float() for o in outs]).mean(0)
        return logits, value


class BossFighter:
    """`policy`, except that the fight rows in boss rooms take `boss`'s logits and value (a boss specialist), in the acts
    `acts` only if set. For evals."""

    def __init__(self, policy, boss, acts=None):
        self.policy, self.boss, self.acts = policy, boss, acts

    def eval(self):
        self.policy.eval()
        self.boss.eval()
        return self

    def train(self, mode=True):
        raise RuntimeError("BossFighter is for evals")

    def __call__(self, ints, floats, practice=None):
        logits, value = self.policy(ints, floats, practice)
        boss = fight_rows(ints) & (ints[:, ROOM_AT] == BOSS_ROOM)
        if self.acts:
            boss &= torch.isin(floats[:, ACT_AT].round().long(), torch.tensor(self.acts, device=ints.device))
        rows = torch.nonzero(boss).squeeze(1)
        if len(rows):
            lg, v = self.boss(ints[rows], floats[rows], practice[rows] if practice is not None else None)
            logits, value = logits.clone(), value.clone()
            logits[rows], value[rows] = lg.to(logits.dtype), v.to(value.dtype)
        return logits, value


class CardPicker:
    """`policy`, except that the card reward rows at floor `from_floor` or later (up to `to_floor`, if given) take `cards`' logits and
    value (its picks and skips). The fights, the rewards screen before a card reward and every other choice stay `policy`'s. For evals."""

    def __init__(self, policy, cards, from_floor=1, to_floor=None):
        self.policy, self.cards, self.from_floor, self.to_floor = policy, cards, from_floor, to_floor

    def eval(self):
        self.policy.eval()
        self.cards.eval()
        return self

    def train(self, mode=True):
        raise RuntimeError("CardPicker is for evals")

    def rows(self, ints, floats):
        """Which rows take `cards`' logits."""
        rows = (ints[:, SCREEN_AT] == CARD_REWARD) & (floats[:, FLOOR_AT] >= self.from_floor)
        return rows if self.to_floor is None else rows & (floats[:, FLOOR_AT] <= self.to_floor)

    def __call__(self, ints, floats, practice=None):
        logits, value = self.policy(ints, floats, practice)
        rows = torch.nonzero(self.rows(ints, floats)).squeeze(1)
        if len(rows):
            lg, v = self.cards(ints[rows], floats[rows], practice[rows] if practice is not None else None)
            logits, value = logits.clone(), value.clone()
            logits[rows], value[rows] = lg.to(logits.dtype), v.to(value.dtype)
        return logits, value


def deck_archetypes(ints):
    """Each row's deck's archetype (analysis.archetypes.arch), from its "deck" field: a card as its id + 1, 0 = none; the first
    MAX_DECK (80) cards of a longer deck."""
    return [arch(LAYOUT["card_java_ids"][c - 1] for c in row if c) for row in ints[:, DECK_AT:DECK_END].tolist()]


def card_reward_moves(ints):
    """The legal moves of card reward rows and, of them, the cards, each (rows, ACTIONS) bool. A policy isn't handed the mask, and
    here the row says it all (encode_run.rs): every option is legal (the cards, then a Singing Bowl's) and so is the skip."""
    kinds = ints[:, OPT_KIND_AT:OPT_KIND_END]
    legal = torch.zeros(len(ints), ACTIONS, dtype=torch.bool, device=ints.device)
    cards = legal.clone()
    legal[:, PICK_AT:PICK_AT + kinds.shape[1]] = kinds != 0
    legal[:, CONFIRM] = True
    cards[:, PICK_AT:PICK_AT + kinds.shape[1]] = kinds == PICK_CARD
    return legal, cards


def max_default(logits, legal, cards):
    """ArchetypeRouter's "max": what the pickers' `logits` (pickers, rows, ACTIONS) want between them. Each one's softmax over the
    `legal` moves; a card (in `cards`) takes its largest probability over the pickers and any other move, the skip above all, its
    smallest (so a skip only if every picker wants to skip); renormalised, as logits. An archetype's defining card is taken when
    offered, and once it's in the deck the run is routed to that archetype's picker: runs commit to what they find."""
    p = torch.softmax(logits.masked_fill(~legal, float("-inf")), -1)
    q = torch.where(cards, p.amax(0), p.amin(0))
    return (q / q.sum(-1, keepdim=True)).log().clamp_min(-1e9)


class ArchetypeRouter:
    """`policy`, except that the card reward rows at floor `from_floor` or later go by the archetype of that run's deck as it stands:
    `pickers`[archetype] (a network, under analysis.archetypes' names) takes the row's logits and value, as CardPicker's `cards` does.
    A deck whose archetype has no picker (at first most decks: attacks or mixed) goes by `default`: "base" leaves the row to `policy`,
    "max" takes the pickers' logits together (max_default), with `policy`'s value. The fights and every other choice stay `policy`'s.
    For evals."""

    def __init__(self, policy, pickers, from_floor=1, default="max"):
        assert default in ("base", "max")
        self.policy, self.pickers, self.from_floor, self.default = policy, dict(pickers), from_floor, default

    def eval(self):
        self.policy.eval()
        for net in self.pickers.values():
            net.eval()
        return self

    def train(self, mode=True):
        raise RuntimeError("ArchetypeRouter is for evals")

    def deciders(self, ints, floats):
        """Who decides at each card reward row from `from_floor` on: {row: its deck's archetype, or "max"}. A row not in it stays
        `policy`'s."""
        rows = torch.nonzero((ints[:, SCREEN_AT] == CARD_REWARD) & (floats[:, FLOOR_AT] >= self.from_floor)).squeeze(1)
        who = {}
        for g, a in zip(rows.tolist(), deck_archetypes(ints[rows])):
            if a in self.pickers:
                who[g] = a
            elif self.default == "max":
                who[g] = "max"
        return who

    def rows(self, ints, floats):
        """Which rows take a picker's logits (a row left to `policy` isn't one)."""
        rows = torch.zeros(len(ints), dtype=torch.bool, device=ints.device)
        rows[torch.tensor(list(self.deciders(ints, floats)), dtype=torch.long, device=ints.device)] = True
        return rows

    def __call__(self, ints, floats, practice=None):
        logits, value = self.policy(ints, floats, practice)
        who = self.deciders(ints, floats)
        if not who:
            return logits, value
        logits, value = logits.float().clone(), value.clone()

        def ask(net, rows):
            return net(ints[rows], floats[rows], practice[rows] if practice is not None else None)

        for name, net in self.pickers.items():
            rows = [g for g, w in who.items() if w == name]
            if rows:
                lg, v = ask(net, rows)
                logits[rows], value[rows] = lg.float(), v.to(value.dtype)
        rows = [g for g, w in who.items() if w == "max"]
        if rows:
            logits[rows] = max_default(torch.stack([ask(net, rows)[0].float() for net in self.pickers.values()]), *card_reward_moves(ints[rows]))
        return logits, value
