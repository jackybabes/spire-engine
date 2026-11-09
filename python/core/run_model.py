"""The run network: the fight network plus the run's context and the choices between fights.

Every row gets a run token (screen, act boss, room, event, grid purpose, HP, gold, floor, act,
keys...). In combat the row is the fight's tokens plus the run token, and actions come from the
fight network's heads, so a trained fight network carries over unchanged. Outside combat the
tokens are the run token, the relics (pooled), the potions, the deck (one token per card) and
the screen's options; pick(k) is scored from option token k, confirm (proceed/skip) from the run
token, and drinking or discarding potion p from potion token p. The value always comes from the
run token, so it estimates how the whole run will go.

The unblended network (`RunNet(unblended=True)`, fight_model.py) has a token per relic outside
combat too (after the deck, the relics token masked out).

With `outcome=True` the network has a human-outcome head (`OutcomeHead`, `outcome_pred`): from a
position between fights, how a human's run went from there (floors still climbed, won, beat the
Heart), read from the run token and trained on human runs (`pretrain_human.py --outcome`,
`train_runs.py --human-outcome`). It never feeds the PPO value: it teaches the shared network what
makes a run strong. Releases leave it out (`export_release.py`).

With `next_move=True` the network has a next-move head (`move_head`, `next_move_pred`; Jaxcalibur's
opponent's-next-action loss, notes/ideas-from-other-game-ais.md): from each monster token in a fight,
the move the monster makes after the one its intent shows, as its place among its monster's moves,
trained on labels from the rollout (`train_runs.py --next-move`). It feeds nothing at play time;
it teaches the fight part the monsters' patterns. Releases leave it out too.

With `fight_outcome=True` the network has a fight-outcome head (`FightOutcomeHead`,
`fight_outcome_pred`; jack, 2026-10-30: "add the output head that gives a score of whether the network
thinks it will win a fight"): from any fight position, the chance the network's own play wins the
practice fight (an A20 Act 3 scenario: both its bosses) and the share of max HP left at its end (0
if lost), from the run token, trained on how the rollout's practice fights ended (`train_runs.py
--fight-outcome`). It reads the encoded token detached, so it never changes how the network plays:
a reading for the deck network ("with this deck, my chance against this boss"). Releases leave it out.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.fight_model import (
    ACTIONS, HAND_FEATS, MAX_MONSTERS, MAX_OPTIONS, MAX_ORBS, MAX_POTIONS, MONSTERS_AT, MOVES_SEEN, PAIR_FEATS, SCALARS, TARGETS, FightNet,
    scale, pair_features)
from core.sts_env import FLOAT_LEN, INT_LEN, LAYOUT

V = LAYOUT["vocab"]
A = LAYOUT["actions"]
PICK = A["pick"][0]
CONFIRM = A["confirm"]
POTION = A["potion"][0]
COMBAT = 1 + LAYOUT["screens"].index("combat")


def _offsets(fields, start):
    out, at = {}, start
    for name, n in fields:
        out[name] = (at, at + n)
        at += n
    return out


RUN_INT_AT = _offsets(LAYOUT["run_ints"], INT_LEN)
RUN_FLOAT_AT = _offsets(LAYOUT["run_floats"], FLOAT_LEN)
RUN_SCALARS = RUN_FLOAT_AT["run_scalars"][1] - RUN_FLOAT_AT["run_scalars"][0]
MAX_DECK = RUN_INT_AT["deck"][1] - RUN_INT_AT["deck"][0]
OPT_FEATS = (RUN_FLOAT_AT["opt"][1] - RUN_FLOAT_AT["opt"][0]) // MAX_OPTIONS
# The deck network's lookup (encoding 3; core/lookup.py): numbers per option ("opt_lookup") and for the run ("run_lookup").
LOOKUP_FEATS = RUN_FLOAT_AT["run_lookup"][1] - RUN_FLOAT_AT["run_lookup"][0]
EVENT_OPTION = 1 + LAYOUT["option_kinds"].index("event_option")


# The outcome head's floors still to climb are divided by this (a run has 57 floors at most).
OUTCOME_FLOORS = 20.0


class OutcomeHead(nn.Module):
    """How a human's run went from a position between fights: floors still climbed (divided by
    OUTCOME_FLOORS), and as logits whether the run won and whether it beat the Heart."""

    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 3))

    def forward(self, g):
        return self.mlp(g)


def outcome_loss(pred, outcome):
    """pred (B, 3) from `RunNet.outcome_pred`; outcome (B, 4): the decision's floor, the floor the
    run ended on, won, beat the Heart (`human_data.Human`) -> (loss, stats): the floors' squared
    error plus both cross-entropies."""
    pred = pred.float()
    floors = (outcome[:, 1] - outcome[:, 0]).clamp(min=0) / OUTCOME_FLOORS
    mse = F.mse_loss(pred[:, 0], floors)
    won = F.binary_cross_entropy_with_logits(pred[:, 1], outcome[:, 2])
    heart = F.binary_cross_entropy_with_logits(pred[:, 2], outcome[:, 3])
    with torch.no_grad():
        acc = ((pred[:, 1] > 0).float() == outcome[:, 2]).float().mean()
    return mse + won + heart, {"outcome_floors": mse.item(), "outcome_won": won.item(), "outcome_heart": heart.item(),
                               "outcome_won_acc": acc.item()}


class FightOutcomeHead(nn.Module):
    """From a fight's encoded run token: as logits, whether the network's play wins the fight, and the
    share of max HP left at its end (0 if lost; through a sigmoid)."""

    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2))

    def forward(self, g):
        return self.mlp(g)


def fight_outcome_loss(pred, won, hp):
    """pred (B, 2) from `RunNet.fight_outcome_pred`; won (B,) 0/1 and hp (B,) the share of max HP left
    (0 if lost) -> (loss, stats): the win's cross-entropy plus the HP share's squared error; the share
    of wins and losses called right and the win chance's Brier score."""
    pred = pred.float()
    ce = F.binary_cross_entropy_with_logits(pred[:, 0], won)
    mse = F.mse_loss(torch.sigmoid(pred[:, 1]), hp)
    with torch.no_grad():
        p = torch.sigmoid(pred[:, 0])
        acc = ((p > 0.5).float() == won).float().mean()
        brier = (p - won).pow(2).mean()
    return ce + mse, {"fight_outcome_ce": ce, "fight_outcome_hp_err": mse, "fight_outcome_acc": acc, "fight_outcome_brier": brier}


# A monster's moves in the move vocabulary (encode.rs `MOVES_PER_MONSTER`): id 1 + monster * 16 + move.
MOVES_PER_MONSTER = 16


class RunNet(FightNet):
    def __init__(self, d=128, layers=3, heads=4, unblended=False, outcome=False, next_move=False, fight_outcome=False):
        super().__init__(d, layers, heads, unblended)
        self.screen = nn.Embedding(V["screen"], d)
        self.encounter = nn.Embedding(V["encounter"], d)
        self.room = nn.Embedding(V["room"], d)
        self.event = nn.Embedding(V["event"], d)
        self.event_screen = nn.Embedding(V["event_screen"], d)
        self.event_index = nn.Embedding(V["event_index"], d)
        self.grid_purpose = nn.Embedding(V["grid_purpose"], d)
        self.neow = nn.Embedding(V["neow"], d)
        self.neow_drawback = nn.Embedding(V["neow_drawback"], d)
        self.option_kind = nn.Embedding(V["option_kind"], d)
        self.run_kinds = nn.Embedding(5, d)  # token type: run, relics, potions, deck, options
        self.run_scalars = nn.Linear(2 * RUN_SCALARS, d)
        self.run_option_feats = nn.Linear(2 * OPT_FEATS, d)
        # The lookup (core/lookup.py) into the option tokens and the run token between fights; at zero, so a network
        # plays as before until training gives the lookup a use (older checkpoints get zeros too: _new_weights).
        self.lookup_opt = nn.Linear(LOOKUP_FEATS, d)
        self.lookup_run = nn.Linear(LOOKUP_FEATS, d)
        for lin in (self.lookup_opt, self.lookup_run):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        # And straight onto each option's pick logit: the screen's weights times the option's four lookup numbers (at zero
        # too). A few weights per screen that every choice there trains, so the network can learn quickly whether to
        # follow the lookup (the map's numbers, win chance and HP left rather than gains, get weights of their own).
        self.lookup_pick = nn.Embedding(V["screen"], LOOKUP_FEATS)
        nn.init.zeros_(self.lookup_pick.weight)

        self.run_pick = nn.Linear(d, 1)
        self.run_confirm = nn.Linear(d, 1)
        self.run_potion = nn.Linear(d, 2)  # drink, discard
        self.run_value = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        # Practice fights (training only) end with the fight and score how it went, not the run:
        # their own value head (same input), so the run value isn't asked two things.
        self.practice_value = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.outcome = OutcomeHead(d) if outcome else None
        self.move_head = nn.Linear(d, MOVES_PER_MONSTER) if next_move else None
        self.fight_outcome_head = FightOutcomeHead(d) if fight_outcome else None

    def _run_token(self, I, Fl):
        return (
            self.screen(I("screen")[:, 0])
            + self.encounter(I("boss")[:, 0])
            + self.room(I("room")[:, 0])
            + self.event(I("event")[:, 0])
            + self.event_screen(I("event_screen")[:, 0])
            + self.grid_purpose(I("grid_purpose")[:, 0])
            + self.run_scalars(scale(Fl("run_scalars")))
            + self.run_kinds.weight[0]
        )

    def forward(self, ints, floats, practice=None):
        """ints (B, RUN_INTS) int64, floats (B, RUN_FLOATS) -> logits (B, ACTIONS), value (B,).
        `practice` (B,) bool marks practice-fight rows, whose value comes from `practice_value`."""
        B = ints.shape[0]
        combat = ints[:, RUN_INT_AT["screen"][0]] == COMBAT
        logits = torch.zeros(B, ACTIONS, device=ints.device)
        value = torch.zeros(B, device=ints.device)
        for rows, fn in ((torch.nonzero(combat).squeeze(1), self._combat), (torch.nonzero(~combat).squeeze(1), self._choices)):
            if rows.numel():
                if fn == self._combat:
                    lg, v = fn(ints[rows], floats[rows], None if practice is None else practice[rows])
                else:
                    lg, v = fn(ints[rows], floats[rows])
                logits[rows] = lg.to(logits.dtype)
                value[rows] = v.to(value.dtype)
        return logits, value

    def _fields(self, ints, floats):
        def I(name):
            a, b = RUN_INT_AT[name]
            return ints[:, a:b]

        def Fl(name):
            a, b = RUN_FLOAT_AT[name]
            return floats[:, a:b]

        return I, Fl

    def _combat_encoded(self, ints, floats):
        """The fight's tokens with the run token, encoded: (B, tokens, d) and the run token's place."""
        I, Fl = self._fields(ints, floats)
        tokens, pad, n_options = self.fight_tokens(ints[:, :INT_LEN], floats[:, :FLOAT_LEN])
        run = self._run_token(I, Fl)
        # The run token goes just before the option tokens (which _encode trims), after the orbs.
        at = tokens.shape[1] - MAX_OPTIONS
        tokens = torch.cat([tokens[:, :at], run[:, None], tokens[:, at:]], dim=1)
        no_pad = torch.zeros_like(pad[:, :1])
        pad = torch.cat([pad[:, :at], no_pad, pad[:, at:]], dim=1)
        return self.out_norm(self._encode(tokens, pad, n_options)), at

    def _combat(self, ints, floats, practice=None):
        x, at = self._combat_encoded(ints, floats)
        g = x[:, at]
        value = self.run_value(g).squeeze(-1)
        if practice is not None and bool(practice.any()):
            value = torch.where(practice, self.practice_value(g).squeeze(-1).to(value.dtype), value)
        pair = pair_features(floats[:, :FLOAT_LEN])
        return self.fight_logits(x, options_at=at + 1, pair=pair), value

    def combat_static(self, ints, floats, practice, options, orbs=True):
        """The combat pass with fixed shapes, so it can be captured as a CUDA graph: every orb
        slot's token (or, with `orbs` False, none: for rows with no orb, whose orb tokens are all
        masked, 2026-10-27) and the unblended network's pieces up to their caps (the empty ones
        masked, where `_combat` leaves them out) and all 64 option tokens (`options`, for rows with
        a selection pending) or none (for rows without one; their option outputs are zeros, as
        `_encode` leaves them). Every row must be in combat. Both value heads are computed and
        `practice` (B,) bool picks. The same logits and value as `_combat`."""
        I, Fl = self._fields(ints, floats)
        tokens, pad, _ = self.fight_tokens(ints[:, :INT_LEN], floats[:, :FLOAT_LEN], fixed=True, orbs=orbs, options=options)
        run = self._run_token(I, Fl)
        at = tokens.shape[1] - (MAX_OPTIONS if options else 0)
        tokens = torch.cat([tokens[:, :at], run[:, None], tokens[:, at:]], dim=1)
        pad = torch.cat([pad[:, :at], torch.zeros_like(pad[:, :1]), pad[:, at:]], dim=1)
        x = self.encoder(tokens, src_key_padding_mask=pad)
        if not options:
            x = torch.cat([x, x.new_zeros(x.shape[0], MAX_OPTIONS, x.shape[2])], dim=1)
        x = self.out_norm(x)
        g = x[:, at]
        value = torch.where(practice, self.practice_value(g).squeeze(-1), self.run_value(g).squeeze(-1))
        pair = pair_features(floats[:, :FLOAT_LEN])
        return self.fight_logits(x, options_at=at + 1, pair=pair), value

    def next_move_pred(self, ints, floats):
        """The next-move head's (B, MAX_MONSTERS, MOVES_PER_MONSTER) logits for fight rows: each
        monster token's guess at the move after the one its intent shows."""
        x, _ = self._combat_encoded(ints, floats)
        return self.move_head(x[:, MONSTERS_AT:MONSTERS_AT + MAX_MONSTERS])

    def fight_outcome_pred(self, ints, floats):
        """The fight-outcome head's (B, 2) logits for fight rows (`FightOutcomeHead`): the network's
        chance of winning the fight and the HP share it ends with. The encoded token is detached, so
        training the head leaves the rest of the network as it was."""
        with torch.no_grad():
            x, at = self._combat_encoded(ints, floats)
        return self.fight_outcome_head(x[:, at])

    def fight_outcome_static(self, ints, floats, orbs=True):
        """`fight_outcome_pred` with fixed shapes, as `combat_static` (every orb slot's token, or none with `orbs`
        False; no option tokens: fight rows with no selection pending), so it can be captured as a CUDA graph (the
        deck network's lookup, core/lookup.py). The same outputs as `fight_outcome_pred` on those rows."""
        I, Fl = self._fields(ints, floats)
        tokens, pad, _ = self.fight_tokens(ints[:, :INT_LEN], floats[:, :FLOAT_LEN], fixed=True, orbs=orbs, options=False)
        run = self._run_token(I, Fl)
        at = tokens.shape[1]
        tokens = torch.cat([tokens, run[:, None]], dim=1)
        pad = torch.cat([pad, torch.zeros_like(pad[:, :1])], dim=1)
        x = self.out_norm(self.encoder(tokens, src_key_padding_mask=pad))
        return self.fight_outcome_head(x[:, at])

    def outcome_pred(self, ints, floats):
        """The outcome head's (B, 3) for rows between fights (`OutcomeHead`)."""
        _, _, g = self._choices(ints, floats, token=True)
        return self.outcome(g)

    def choices_static(self, ints, floats, deck_cap, opt_cap):
        """The pass for rows between fights with fixed shapes, so it can be captured as a CUDA
        graph: `deck_cap` deck tokens and `opt_cap` option tokens whatever the rows hold (every row
        must fit; the empty ones masked, where `_choices` leaves out what no row needs). The same
        logits (of the legal moves: an option slot past a row's options is filled differently, and
        masked out either way) and value as `_choices`. The unblended network's relic tokens are all the
        relic slots (32), the empty ones masked (jack, 2026-10-30, the deck network: "compile buffers"; its
        choices had gone through the usual pass, the rollout's GPU about a third busy)."""
        return self._choices(ints, floats, caps=(deck_cap, opt_cap))

    def _choices(self, ints, floats, token=False, caps=None):
        """Logits and value for rows between fights (and, with `token`, the encoded run token).
        `caps` (deck, options): `choices_static`'s fixed token counts."""
        I, Fl = self._fields(ints, floats)
        B = ints.shape[0]
        k = self.run_kinds.weight
        run = self._run_token(I, Fl) + self.lookup_run(Fl("run_lookup"))
        if self.unblended:  # a token per relic, after the deck
            rel_ids = I("run_relic")
            # (With fixed caps, every relic slot, the empty ones masked: no count that waits for the GPU.)
            n_rel = rel_ids.shape[1] if caps else max(int((rel_ids > 0).sum(1).max()), 1)
            rel_ids = rel_ids[:, :n_rel]
            relic_pieces = self._amount_tokens(self.relic, rel_ids, Fl("run_relic_counter")[:, :n_rel]) + k[1]
            relics = k[1].expand(B, -1)
        else:
            relics = self._amounts(self.relic, I("run_relic"), Fl("run_relic_counter")) + k[1]
        pot_ids = I("run_potion")
        pots = self.potion(pot_ids) + k[2]

        # Only as many deck and option tokens as the longest row needs (or the fixed caps).
        deck_ids = I("deck")
        n_deck = caps[0] if caps else max(int((deck_ids > 0).sum(1).max()), 1)
        deck_ids = deck_ids[:, :n_deck]
        deck = self.card(deck_ids) + self.upgraded(Fl("deck_upgrades")[:, :n_deck, None]) + k[3]

        kinds = I("opt_kind")
        n_opt = caps[1] if caps else max(int((kinds > 0).sum(1).max()), 1)
        kinds = kinds[:, :n_opt]

        def O(name):
            return I(name)[:, :n_opt]

        feats = Fl("opt").reshape(B, MAX_OPTIONS, OPT_FEATS)[:, :n_opt]
        is_event = (kinds == EVENT_OPTION)[..., None]
        opts = (
            self.option_kind(kinds)
            + self.card(O("opt_card"))
            + self.relic(O("opt_relic"))
            + self.potion(O("opt_potion"))
            + self.neow(O("opt_neow"))
            + self.neow_drawback(O("opt_drawback"))
            + self.room(O("opt_room"))
            + self.event_index(O("opt_index"))
            + self.event(I("event")) * is_event
            + self.run_option_feats(scale(feats))
            + self.lookup_opt(Fl("opt_lookup").reshape(B, MAX_OPTIONS, LOOKUP_FEATS)[:, :n_opt])
            + k[4]
        )

        first = torch.zeros(B, 2, dtype=torch.bool, device=ints.device)
        tokens = [run[:, None], relics[:, None], pots, deck]
        pad = [first, pot_ids == 0, deck_ids == 0]
        if self.unblended:
            first[:, 1] = True  # the relics token: empty
            tokens.append(relic_pieces)
            pad.append(rel_ids == 0)
        tokens = torch.cat(tokens + [opts], dim=1)
        pad = torch.cat(pad + [kinds == 0], dim=1)
        x = self.out_norm(self.encoder(tokens, src_key_padding_mask=pad))
        g = x[:, 0]
        p = x[:, 2:2 + MAX_POTIONS]
        o = x[:, tokens.shape[1] - n_opt:]

        logits = torch.zeros(B, ACTIONS, dtype=x.dtype, device=x.device)
        nudge = (Fl("opt_lookup").reshape(B, MAX_OPTIONS, LOOKUP_FEATS)[:, :n_opt] * self.lookup_pick(I("screen")[:, 0])[:, None]).sum(-1)
        logits[:, PICK:PICK + n_opt] = self.run_pick(o).squeeze(-1) + nudge.to(x.dtype)
        logits[:, CONFIRM] = self.run_confirm(g).squeeze(-1)
        pot = self.run_potion(p)  # (B, MAX_POTIONS, 2)
        for t in range(2):
            logits[:, POTION + t:POTION + MAX_POTIONS * TARGETS:TARGETS] = pot[:, :, t]
        if token:
            return logits, self.run_value(g).squeeze(-1), g
        return logits, self.run_value(g).squeeze(-1)


# Option kinds added after a checkpoint was saved, and the old kind each new one starts as (so
# the network behaves as before until training tells them apart).
NEW_KIND_FROM = {k: "grid_card" for k in ("grid_remove", "grid_upgrade", "grid_transform", "grid_duplicate", "grid_bottle", "grid_offer")}


# Weights added since a checkpoint was saved. Those feeding tokens it already had start at zero,
# so the network plays as before until training uses them (the stance and the last card type on
# the global and player tokens; the projection of a monster's last moves); the moves' embeddings
# behind that projection and the orb tokens' own weights start fresh (from a fixed seed, so every
# load of a checkpoint builds the same network).
def _new_weights(d):
    return {
        "stance.weight": ("zero", (V["stance"], d)),
        "card_type.weight": ("zero", (V["card_type"], d)),
        "moves_seen.weight": ("zero", (d, MOVES_SEEN * d)),
        "move.weight": ("embedding", (V["move"], d)),
        "orb.weight": ("embedding", (V["orb"], d)),
        "orb_slot.weight": ("embedding", (MAX_ORBS, d)),
        "orb_feats.weight": ("linear", (d, 4)),
        "orb_feats.bias": ("linear", (d,)),
        "lookup_opt.weight": ("zero", (d, LOOKUP_FEATS)),
        "lookup_opt.bias": ("zero", (d,)),
        "lookup_run.weight": ("zero", (d, LOOKUP_FEATS)),
        "lookup_run.bias": ("zero", (d,)),
        "lookup_pick.weight": ("zero", (V["screen"], LOOKUP_FEATS)),
    }


def _grow_scaled(w, n, keep):
    """A Linear's weight over scale()d features (a symlog block, then an x / 100 block) grown to
    `n` features a block: the first `keep` of each old block kept, the rest at zero."""
    old = w.shape[1] // 2
    grown = torch.zeros(w.shape[0], 2 * n, dtype=w.dtype, device=w.device)
    grown[:, :keep] = w[:, :keep]
    grown[:, n:n + keep] = w[:, old:old + keep]
    return grown


def upgrade_state(state):
    """Brings an older run checkpoint's weights up to the current layout, in place: a smaller
    option-kind embedding grows, each new kind's row copied from the kind it replaces, and the
    practice-fight value head starts as a copy of the run value head. Encoding 2's inputs: the hand
    features keep their first eight columns (Feel No Pain's and Dark Embrace's, 8 and 9 in
    encoding 1, are gone) and the new ones start at zero, as do the new scalars and the play
    scorer's new pair features; the new weights are added (`_new_weights`). So the network plays as
    before until training uses the new inputs, except where Feel No Pain or Dark Embrace mattered
    (and in fights with orbs, whose tokens are new)."""
    for k in [k for k in state if k.startswith("run_value.")]:
        state.setdefault("practice_value." + k[len("run_value."):], state[k].clone())
    w = state.get("hand_feats.weight")
    if w is not None and w.shape[1] != 2 * HAND_FEATS:
        state["hand_feats.weight"] = _grow_scaled(w, HAND_FEATS, min(w.shape[1] // 2, 8))
    w = state.get("scalars.weight")
    if w is not None and w.shape[1] != 2 * SCALARS:
        state["scalars.weight"] = _grow_scaled(w, SCALARS, w.shape[1] // 2)
    a = state.get("play.a.weight")
    if a is not None:
        c = state.get("play.c.weight", torch.zeros(a.shape[0], 0, dtype=a.dtype, device=a.device))
        if c.shape[1] != PAIR_FEATS:
            grown = torch.zeros(a.shape[0], PAIR_FEATS, dtype=a.dtype, device=a.device)
            grown[:, :c.shape[1]] = c
            state["play.c.weight"] = grown
    card = state.get("card.weight")
    if card is not None:
        gen = torch.Generator().manual_seed(1)
        for key, (how, shape) in _new_weights(card.shape[1]).items():
            if key in state:
                continue
            if how == "zero":
                x = torch.zeros(shape)
            elif how == "embedding":
                x = torch.randn(shape, generator=gen) * card.float().std().cpu()
            else:  # nn.Linear's default: uniform within 1 / sqrt(fan in)
                bound = 1.0 / (4 ** 0.5)
                x = (torch.rand(shape, generator=gen) * 2 - 1) * bound
            state[key] = x.to(card.dtype).to(card.device)
    w = state.get("option_kind.weight")
    if w is None or w.shape[0] >= V["option_kind"]:
        return state
    kinds = ["-"] + LAYOUT["option_kinds"]
    grown = torch.empty(V["option_kind"], w.shape[1], dtype=w.dtype, device=w.device)
    grown[:w.shape[0]] = w
    for i in range(w.shape[0], V["option_kind"]):
        grown[i] = w[kinds.index(NEW_KIND_FROM[kinds[i]])]
    state["option_kind.weight"] = grown
    return state


def from_fight_checkpoint(path, device):
    """A RunNet whose fight parts start from a trained fight network's weights."""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    net = RunNet(d=a["d"], layers=a["layers"], heads=a["heads"], unblended=a.get("unblended", False),
                 outcome=a.get("outcome", False)).to(device)
    missing, unexpected = net.load_state_dict(ck["net"], strict=False)
    assert not unexpected, unexpected
    return net
