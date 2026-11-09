"""The fight network: a small transformer over the fight's tokens, with a policy over the 171
action slots and a value head.

Tokens (d wide): one global token (scalars, character, pending choice, the last card type
played), one player token (scalars, stance, pooled powers), one relic token (pooled), three pile
tokens (draw, discard, exhaust; pooled cards), 10 hand cards, 6 monsters (with pooled powers and
their last two moves), 5 potions, the orb slots (up to 10, each with its position: 0 is the
front, the next evoked; only as many as the batch's most) and 64 selection options. Empty slots
are masked out of attention.

Unblended (`FightNet(unblended=True)`, 2026-10-27; notes/watcher-training-plan.md "no blending"):
the relics, the powers and the piles aren't summed into the player, relic, monster and pile
tokens; each is a piece (token) of its own, so a card or an option can look at one particular
relic, power or pile card. A piece per relic (with its counter), per power (with its amount and
whose it is: the player's, or monster k's, the same vector the monster's token carries) and per
distinct pile card (card and upgrades, with how many copies are in the draw, discard and exhaust
piles), after the orbs. The relic and pile tokens are masked out then, and the player and monster
tokens carry no powers.

Action logits come from the tokens they refer to: play(h, t) scores hand token h against target
t (t = 0 a learned "no target" vector, else monster token t - 1); potion(p, t) the same with
potion tokens; pick(k) from option token k; end turn, confirm and the value from the global
token. So a card is judged the same way wherever it sits in the hand.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.sts_env import LAYOUT

V = LAYOUT["vocab"]
A = LAYOUT["actions"]
MAX_HAND = A["play"][1]
TARGETS = A["play"][2]
MAX_MONSTERS = TARGETS - 1
MAX_POTIONS = A["potion"][1]
MAX_OPTIONS = A["pick"][1]
ACTIONS = A["count"]


def _offsets(fields):
    out, at = {}, 0
    for name, n in fields:
        out[name] = (at, at + n)
        at += n
    return out


INT_AT = _offsets(LAYOUT["ints"])
FLOAT_AT = _offsets(LAYOUT["floats"])
MAX_PILE = INT_AT["draw_pile"][1] - INT_AT["draw_pile"][0]
MAX_MONSTER_POWERS = (INT_AT["monster_power"][1] - INT_AT["monster_power"][0]) // MAX_MONSTERS
HAND_FEATS = (FLOAT_AT["hand"][1] - FLOAT_AT["hand"][0]) // MAX_HAND
MONSTER_FEATS = (FLOAT_AT["monster"][1] - FLOAT_AT["monster"][0]) // MAX_MONSTERS
OPTION_FEATS = (FLOAT_AT["option"][1] - FLOAT_AT["option"][0]) // MAX_OPTIONS
SCALARS = FLOAT_AT["scalars"][1] - FLOAT_AT["scalars"][0]
MAX_ORBS = INT_AT["orb"][1] - INT_AT["orb"][0]
MOVES_SEEN = (INT_AT["monster_moves"][1] - INT_AT["monster_moves"][0]) // MAX_MONSTERS
# Fight tokens before the orbs and the options: global, player, relics, 3 piles, hand, monsters,
# potions. The orb tokens follow (as many as the batch needs, `fight_tokens`).
FIGHT_FIXED = 6 + MAX_HAND + MAX_MONSTERS + MAX_POTIONS
# Where the monster tokens start (after the global, player, relic, 3 pile and hand tokens).
MONSTERS_AT = 6 + MAX_HAND
# Hand row columns (encode.rs `hand_row`): every enemy, random enemies.
HAND_ALL, HAND_RANDOM = 9, 10
# The unblended network's pieces a fight: at most this many relics, powers (the player's first, then
# the monsters') and distinct pile cards; a fight with more leaves the last ones out. Sized from the
# shallow network's Watcher A20 runs (2026-10-27, 25,564 fight positions at floor 28 on average): at
# most 14 relics, 8 + 10 powers and 25 distinct pile cards.
MAX_RELIC_PIECES, MAX_POWER_PIECES, MAX_CARD_PIECES = 24, 24, 40
PILES = ("draw_pile", "discard_pile", "exhaust_pile")
MAX_UPGRADES = 127  # a pile card's upgrades (Searing Blow's are unbounded) as told apart in its piece


def scale(x):
    """Raw game numbers (HP, block, counters, ...) as two features each: symlog and x / 100."""
    return torch.cat([torch.sign(x) * torch.log1p(x.abs()), x / 100.0], dim=-1)


class Scorer(nn.Module):
    """Additive attention score of each of `a` against each of `b`: (B, n, d) x (B, m, d) -> (B, n, m).
    With `pair` features (B, n, m, pair_dim) for each (a, b) pair, they join the sum (starting at
    zero, so a checkpoint from before them behaves as it did)."""

    def __init__(self, d, pair_dim=0):
        super().__init__()
        self.a = nn.Linear(d, d)
        self.b = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, 1)
        self.c = None
        if pair_dim:
            self.c = nn.Linear(pair_dim, d, bias=False)
            nn.init.zeros_(self.c.weight)

    def forward(self, a, b, pair=None):
        x = self.a(a)[:, :, None, :] + self.b(b)[:, None, :, :]
        if self.c is not None and pair is not None:
            x = x + self.c(pair).to(x.dtype)
        return self.v(torch.tanh(x)).squeeze(-1)


# What playing hand card h at target t would do. For t = 1-6 (a monster): one hit's damage to it,
# whether that kills, and the share of its HP and block it removes; then the same four for the
# whole play (all its hits, `hand_target_total`). For t = 0 (no target), an attack on every enemy
# or on random ones: zeros, then the damage that lands summed over the monsters (each capped at
# its HP and block, scaled as two numbers), how many it kills and whether it kills them all.
PAIR_FEATS = 8


def _hit_features(dmg, ehp):
    kills = ((dmg >= ehp) & (ehp > 0) & (dmg > 0)).float()
    share = torch.where(ehp > 0, (dmg / ehp.clamp(min=1)).clamp(max=2.0), torch.zeros_like(dmg))
    return torch.cat([scale(dmg[..., None]), kills[..., None], share[..., None]], dim=-1)


def pair_features(floats):
    """(B, MAX_HAND, TARGETS, PAIR_FEATS) from a fight row's floats."""
    B = floats.shape[0]

    def per_card(name):
        a, b = FLOAT_AT[name]
        return floats[:, a:b].reshape(B, MAX_HAND, MAX_MONSTERS)

    dmg, total = per_card("hand_target_damage"), per_card("hand_target_total")
    ma, mb = FLOAT_AT["monster"]
    mon = floats[:, ma:mb].reshape(B, MAX_MONSTERS, -1)
    ehp = (mon[:, :, 0] + mon[:, :, 2])[:, None, :]  # hp + block
    targeted = torch.cat([_hit_features(dmg, ehp), _hit_features(total, ehp)], dim=-1)  # (B, H, M, 8)

    ha, hb = FLOAT_AT["hand"]
    hand = floats[:, ha:hb].reshape(B, MAX_HAND, HAND_FEATS)
    spread = ((hand[:, :, HAND_ALL] > 0) | (hand[:, :, HAND_RANDOM] > 0)).float()  # (B, H)
    present = (ehp > 0).float()
    landed = (torch.minimum(total, ehp.expand_as(total)) * present).sum(-1)
    killed = (((total >= ehp) & (ehp > 0) & (total > 0)).float()).sum(-1)
    all_dead = ((killed >= present.sum(-1)) & (present.sum(-1) > 0)).float()
    none = torch.cat([scale(landed[..., None]), killed[..., None], all_dead[..., None]], dim=-1) * spread[..., None]
    no_target = torch.cat([torch.zeros_like(none), none], dim=-1)[:, :, None, :]  # (B, H, 1, 8)
    return torch.cat([no_target, targeted], dim=2)


def grow_embeddings(module, state, prefix, *_):
    """Load hook: an older checkpoint's embedding tables (cards, powers, ...) get the rows of ids
    added since, which are appended to the vocabularies so the old rows keep their meaning. The
    new rows start like a fresh nn.Embedding (normal, the old rows' spread), from a fixed seed so
    every load of a checkpoint builds the same network."""
    gen = torch.Generator().manual_seed(0)
    for name, emb in module.named_children():
        key = prefix + name + ".weight"
        w = state.get(key)
        if not isinstance(emb, nn.Embedding) or w is None or w.shape[0] >= emb.num_embeddings:
            continue
        extra = torch.randn(emb.num_embeddings - w.shape[0], w.shape[1], generator=gen) * w.float().std().cpu()
        state[key] = torch.cat([w, extra.to(w.dtype).to(w.device)])


def _compact(present, cap, *values):
    """Each row's present entries moved to the front in order, (B, n) -> (B, cap) each, the rest
    zero; entries past `cap` left out."""
    pos = present.long().cumsum(1) - 1
    at = torch.where(present & (pos < cap), pos, torch.full_like(pos, cap))
    return [v.new_zeros(v.shape[0], cap + 1).scatter_(1, at, v)[:, :cap] for v in values]


def piece_ids(ints, floats, fixed=False):
    """What the unblended network's pieces are, from a fight row: the relics (ids, counters; in the
    order they were obtained), the powers (ids, amounts and whose: 0 the player, k monster k; the
    player's first, then each monster's in its slot's order) and the distinct pile cards (card ids,
    upgrades and copies in the draw, discard and exhaust piles (B, n, 3); by card id, then
    upgrades). Each (B, n), id 0 for an empty piece. As many of each as the batch's most, or with
    `fixed` the caps (MAX_*_PIECES), for a pass of fixed shape."""
    def I(name):
        a, b = INT_AT[name]
        return ints[:, a:b]

    def Fl(name):
        a, b = FLOAT_AT[name]
        return floats[:, a:b]

    relic, relic_counter = I("relic")[:, :MAX_RELIC_PIECES], Fl("relic_counter")[:, :MAX_RELIC_PIECES]

    # Powers: the player's, then each monster's, moved to the front, with whose they are.
    power = torch.cat([I("player_power"), I("monster_power")], 1)
    n_player = INT_AT["player_power"][1] - INT_AT["player_power"][0]
    owner = torch.arange(1 + MAX_MONSTERS, device=ints.device).repeat_interleave(
        torch.tensor([n_player] + [MAX_MONSTER_POWERS] * MAX_MONSTERS, device=ints.device))
    power, power_amount, power_owner = _compact(
        power > 0, MAX_POWER_PIECES, power, torch.cat([Fl("player_power_amount"), Fl("monster_power_amount")], 1),
        owner.expand_as(power))

    # Pile cards: one piece per distinct (card, upgrades), its copies in each pile counted. Sorting
    # the keys puts each card's copies together; a group starts where the key changes.
    ids = torch.cat([I(p) for p in PILES], 1)
    ups = torch.cat([Fl(p + "_upgrades") for p in PILES], 1).round().long().clamp(0, MAX_UPGRADES)
    empty = (V["card"] + 1) * (MAX_UPGRADES + 1)
    key = torch.where(ids > 0, ids * (MAX_UPGRADES + 1) + ups, torch.full_like(ids, empty))
    key, order = key.sort(1)
    pile = torch.arange(len(PILES), device=ids.device).repeat_interleave(MAX_PILE)[order]
    valid = key < empty
    first = valid & (key != torch.cat([torch.full_like(key[:, :1], -1), key[:, :-1]], 1))
    group = first.long().cumsum(1) - 1
    dump = MAX_CARD_PIECES  # the copies of groups past the cap, and the empty slots, land here
    at = torch.where(valid & (group < MAX_CARD_PIECES), group, torch.full_like(group, dump))
    counts = torch.zeros(key.shape[0], (MAX_CARD_PIECES + 1) * len(PILES), device=ids.device)
    counts = counts.scatter_add_(1, at * len(PILES) + pile, valid.float()).reshape(-1, MAX_CARD_PIECES + 1, len(PILES))
    group_key = key.new_zeros(key.shape[0], MAX_CARD_PIECES + 1).scatter_(1, torch.where(first, at, dump), key)
    group_key, counts = group_key[:, :MAX_CARD_PIECES], counts[:, :MAX_CARD_PIECES]
    card = torch.where(counts.sum(-1) > 0, group_key // (MAX_UPGRADES + 1), torch.zeros_like(group_key))
    out = {"relic": relic, "relic_counter": relic_counter, "power": power, "power_amount": power_amount,
           "power_owner": power_owner, "card": card, "card_upgrades": (group_key % (MAX_UPGRADES + 1)).float(),
           "card_counts": counts}
    if not fixed:  # as many as the batch's most
        def most(x):
            return int((x > 0).sum(1).max()) if x.numel() else 0
        n = {"relic": most(relic), "power": most(power), "card": most(card)}
        out = {name: x[:, :n[name.split("_")[0]]] for name, x in out.items()}
    return out


class FightNet(nn.Module):
    def __init__(self, d=128, layers=3, heads=4, unblended=False):
        super().__init__()
        self.d = d
        self.unblended = unblended
        self.card = nn.Embedding(V["card"], d)
        self.power = nn.Embedding(V["power"], d)
        self.monster = nn.Embedding(V["monster"], d)
        self.intent = nn.Embedding(V["intent"], d)
        self.relic = nn.Embedding(V["relic"], d)
        self.potion = nn.Embedding(V["potion"], d)
        self.kind = nn.Embedding(V["choice_kind"], d)
        self.pile = nn.Embedding(V["choice_pile"], d)
        self.character = nn.Embedding(V["character"], d)
        self.stance = nn.Embedding(V["stance"], d)
        self.card_type = nn.Embedding(V["card_type"], d)
        self.orb = nn.Embedding(V["orb"], d)
        self.orb_slot = nn.Embedding(MAX_ORBS, d)
        self.move = nn.Embedding(V["move"], d)
        self.kinds = nn.Embedding(9, d)  # token type: global, player, relics, 3 piles, ... (see forward)

        self.scalars = nn.Linear(2 * SCALARS, d)
        self.hand_feats = nn.Linear(2 * HAND_FEATS, d)
        self.monster_feats = nn.Linear(2 * MONSTER_FEATS, d)
        self.option_feats = nn.Linear(2 * OPTION_FEATS, d)
        self.orb_feats = nn.Linear(4, d)  # passive and evoke, scaled
        self.moves_seen = nn.Linear(MOVES_SEEN * d, d, bias=False)  # a monster's last moves, each in its place
        self.upgraded = nn.Linear(1, d, bias=False)
        # An entity with an amount (power stacks, relic counter): MLP(embedding, scaled amount).
        self.amount = nn.Sequential(nn.Linear(d + 2, d), nn.ReLU(), nn.Linear(d, d))

        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)

        self.no_target = nn.Parameter(torch.zeros(d))
        self.play = Scorer(d, PAIR_FEATS)
        self.use_potion = Scorer(d)
        self.pick = nn.Linear(d, 1)
        self.end_turn = nn.Linear(d, 1)
        self.confirm = nn.Linear(d, 1)
        self.value = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        if unblended:
            self.owner = nn.Embedding(1 + MAX_MONSTERS, d)  # whose: the player, monster k (its token carries it too)
            self.piece_kinds = nn.Embedding(3, d)  # piece type: relic, power, pile card
            self.pile_counts = nn.Linear(2 * len(PILES), d)  # a pile card's copies in each pile, scaled
        self._register_load_state_dict_pre_hook(grow_embeddings, with_module=True)

    def _amount_tokens(self, emb, ids, amounts):
        """Each (id, amount) entity's vector: (..., k) -> (..., k, d)."""
        return self.amount(torch.cat([emb(ids), scale(amounts[..., None])], dim=-1))

    def _amounts(self, emb, ids, amounts):
        """Sum over a list of (id, amount) entities, empty ids skipped: (..., k) -> (..., d)."""
        x = self._amount_tokens(emb, ids, amounts)
        return (x * (ids > 0)[..., None]).sum(-2)

    def _pieces(self, ints, floats, fixed):
        """The unblended network's pieces after the orbs (`piece_ids`): relics, powers, distinct pile
        cards (B, n, d), with their padding mask (True = empty)."""
        k = self.piece_kinds.weight
        p = piece_ids(ints, floats, fixed)
        relics = self._amount_tokens(self.relic, p["relic"], p["relic_counter"]) + k[0]
        powers = self._amount_tokens(self.power, p["power"], p["power_amount"]) + self.owner(p["power_owner"]) + k[1]
        cards = self.card(p["card"]) + self.upgraded(p["card_upgrades"][..., None]) + self.pile_counts(scale(p["card_counts"])) + k[2]
        tokens = torch.cat([relics, powers, cards], dim=1)
        return tokens, torch.cat([p["relic"] == 0, p["power"] == 0, p["card"] == 0], dim=1)

    def _pool_cards(self, ids, upgrades):
        # Mean of card embedding + upgrades x w over the pile, computed as card counts x the
        # embedding table (far cheaper than looking up 80 embeddings per pile).
        keep = (ids > 0).float()
        counts = torch.zeros(ids.shape[0], self.card.num_embeddings, device=ids.device).scatter_add_(1, ids, keep)
        x = counts @ self.card.weight + (upgrades * keep).sum(-1, keepdim=True) * self.upgraded.weight[:, 0]
        return x / keep.sum(-1, keepdim=True).clamp(min=1.0)

    def _encode(self, tokens, pad, n_options):
        """The encoder over each row's tokens. Few rows have selection options (about 4%), so rows
        without them skip the 64 option tokens and rows with them stop after the batch's longest
        option list: the same result as encoding all 91 tokens, at a third of the cost."""
        fixed = tokens.shape[1] - MAX_OPTIONS
        x = torch.zeros_like(tokens)
        has = n_options > 0
        for rows, length in ((torch.nonzero(~has).squeeze(1), fixed), (torch.nonzero(has).squeeze(1), None)):
            if rows.numel() == 0:
                continue
            if length is None:
                length = fixed + int(n_options.max())
            y = self.encoder(tokens[rows, :length], src_key_padding_mask=pad[rows, :length])
            x[rows, :length] = y.to(x.dtype)
        return x

    def forward(self, ints, floats):
        """ints (B, INTS) int64, floats (B, FLOATS) float32 -> logits (B, ACTIONS), value (B,)."""
        tokens, pad, n_options = self.fight_tokens(ints, floats)
        x = self.out_norm(self._encode(tokens, pad, n_options))
        options_at = tokens.shape[1] - MAX_OPTIONS
        return self.fight_logits(x, options_at=options_at, pair=pair_features(floats)), self.value(x[:, 0]).squeeze(-1)

    def fight_tokens(self, ints, floats, fixed=False, orbs=True, options=True):
        """The fight's tokens (B, FIGHT_FIXED + orbs + pieces + MAX_OPTIONS, d), the orb tokens (as
        many as the batch's most filled slots; all MAX_ORBS with `fixed`, the empty ones masked, for
        a pass of fixed shape; none with `orbs` False, for rows with no orb, whose orb tokens are
        all masked anyway), the unblended network's pieces (`_pieces`, none for the blended one)
        then the option tokens last (none with `options` False, for rows without a selection, whose
        option tokens the encoder never reads), with their padding mask (True = empty) and each
        row's option count."""
        B = ints.shape[0]

        def I(name):
            a, b = INT_AT[name]
            return ints[:, a:b]

        def Fl(name):
            a, b = FLOAT_AT[name]
            return floats[:, a:b]

        k = self.kinds.weight
        scal = scale(Fl("scalars"))
        glob = (
            self.character(I("character")[:, 0])
            + self.scalars(scal)
            + self.kind(I("choice_kind")[:, 0])
            + self.pile(I("choice_pile")[:, 0])
            + self.card_type(I("last_played_type")[:, 0])
            + k[0]
        )
        if self.unblended:  # the powers, relics and pile cards are pieces of their own (`_pieces`)
            player = self.scalars(scal) + self.stance(I("stance")[:, 0]) + self.owner.weight[0] + k[1]
            relics = k[2].expand(B, -1)
            piles = k[3].expand(B, len(PILES), -1)
        else:
            player = self._amounts(self.power, I("player_power"), Fl("player_power_amount")) + self.scalars(scal) + self.stance(I("stance")[:, 0]) + k[1]
            relics = self._amounts(self.relic, I("relic"), Fl("relic_counter")) + k[2]
            piles = torch.stack([self._pool_cards(I(p), Fl(p + "_upgrades")) for p in PILES], dim=1) + k[3]

        hand_ids = I("hand_card")
        hand = self.card(hand_ids) + self.hand_feats(scale(Fl("hand").reshape(B, MAX_HAND, HAND_FEATS))) + k[4]

        mon_ids = I("monster")
        if self.unblended:
            mon_pow = self.owner.weight[1:]
        else:
            mon_pow = self._amounts(
                self.power,
                I("monster_power").reshape(B, MAX_MONSTERS, MAX_MONSTER_POWERS),
                Fl("monster_power_amount").reshape(B, MAX_MONSTERS, MAX_MONSTER_POWERS),
            )
        moves = self.move(I("monster_moves").reshape(B, MAX_MONSTERS, MOVES_SEEN)).reshape(B, MAX_MONSTERS, MOVES_SEEN * self.d)
        mons = (
            self.monster(mon_ids)
            + self.intent(I("monster_intent"))
            + self.monster_feats(scale(Fl("monster").reshape(B, MAX_MONSTERS, MONSTER_FEATS)))
            + mon_pow
            + self.moves_seen(moves)
            + k[5]
        )

        pot_ids = I("potion")
        pots = self.potion(pot_ids) + k[6]

        # Orb slots, front first: as many tokens as the batch's most slots (often none).
        orb_ids = I("orb")
        n_orbs = 0 if not orbs else MAX_ORBS if fixed else int((orb_ids > 0).sum(1).max()) if orb_ids.numel() else 0
        orb_ids = orb_ids[:, :n_orbs]
        orb_nums = torch.stack([Fl("orb_passive")[:, :n_orbs], Fl("orb_evoke")[:, :n_orbs]], dim=-1)
        orbs = self.orb(orb_ids) + self.orb_feats(scale(orb_nums)) + self.orb_slot.weight[:n_orbs] + k[8]

        opt_ids = I("option_card")
        if options:
            opts = self.card(opt_ids) + self.option_feats(scale(Fl("option").reshape(B, MAX_OPTIONS, OPTION_FEATS))) + k[7]

        first = torch.zeros(B, 6, dtype=torch.bool, device=ints.device)
        tokens = [glob[:, None], player[:, None], relics[:, None], piles, hand, mons, pots, orbs]
        pad = [first, hand_ids == 0, mon_ids == 0, pot_ids == 0, orb_ids == 0]
        if self.unblended:
            first[:, 2:] = True  # the relic and pile tokens: empty
            pieces, piece_pad = self._pieces(ints, floats, fixed)
            tokens.append(pieces)
            pad.append(piece_pad)
        if options:
            tokens, pad = tokens + [opts], pad + [opt_ids == 0]
        return torch.cat(tokens, dim=1), torch.cat(pad, dim=1), (opt_ids > 0).sum(1)

    def fight_logits(self, x, options_at, pair=None):
        """Action logits from the encoded fight tokens (option tokens from `options_at`)."""
        B = x.shape[0]
        at = 6
        g = x[:, 0]
        h = x[:, at:at + MAX_HAND]
        at += MAX_HAND
        m = x[:, at:at + MAX_MONSTERS]
        at += MAX_MONSTERS
        p = x[:, at:at + MAX_POTIONS]
        o = x[:, options_at:options_at + MAX_OPTIONS]
        targets = torch.cat([self.no_target.expand(B, 1, self.d), m], dim=1)
        return torch.cat(
            [
                self.play(h, targets, pair).reshape(B, MAX_HAND * TARGETS),
                self.use_potion(p, targets).reshape(B, MAX_POTIONS * TARGETS),
                self.end_turn(g),
                self.pick(o).squeeze(-1),
                self.confirm(g),
            ],
            dim=1,
        )


def masked(logits, mask):
    return logits.masked_fill(~mask, -1e9)


def to_tensors(obs, device):
    """FightEnv observation -> (ints, floats, mask) tensors on `device`."""
    ints = torch.from_numpy(obs["ints"].astype(np.int64)).to(device)
    floats = torch.from_numpy(np.ascontiguousarray(obs["floats"])).to(device)
    mask = torch.from_numpy(obs["mask"]).to(device)
    return ints, floats, mask


def policy_fn(net, device, greedy=True):
    """A `baselines.evaluate` policy: obs dict, rng -> action slots."""

    @torch.no_grad()
    def act(obs, rng):
        ints, floats, mask = to_tensors(obs, device)
        logits, _ = net(ints, floats)
        logits = masked(logits, mask)
        if greedy:
            a = logits.argmax(-1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        return a.cpu().numpy()

    return act


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
