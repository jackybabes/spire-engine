"""Checks for the unblended network (`RunNet(unblended=True)`: a token per relic, power and distinct
pile card; fight_model.py), plain asserts, no pytest needed:

    python python/tests/test_unblended.py

`piece_ids` against a plain count of the rows (Defect practice fights with random decks, Watcher
runs), and at its caps (a row with more relics, powers and distinct cards than they hold); the
network's outputs don't depend on the order of the relics or of the cards in a pile; the
fixed-shape pass and FixedShapes give the usual pass's outputs; every new weight gets a gradient;
a checkpoint loads as unblended; and the blended network has none of the new weights."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import tempfile

import torch

from core.fight_model import (FLOAT_AT, INT_AT, MAX_CARD_PIECES, MAX_MONSTER_POWERS, MAX_MONSTERS, MAX_POWER_PIECES,
                              MAX_RELIC_PIECES, PILES, piece_ids)
from core.fixed_shapes import FixedShapes
from core.run_model import COMBAT, RUN_FLOAT_AT, RUN_INT_AT, RunNet
from tests.test_static import positions

NEW = ("owner.weight", "piece_kinds.weight", "pile_counts.weight", "pile_counts.bias")


def field(x, at, name):
    a, b = at[name]
    return x[:, a:b]


def expected(ints, floats, r):
    """Row r's pieces counted the plain way: (relics, powers, cards), each a list."""
    rel = field(ints, INT_AT, "relic")[r].tolist()
    cnt = field(floats, FLOAT_AT, "relic_counter")[r].tolist()
    relics = [(i, c) for i, c in zip(rel, cnt) if i > 0][:MAX_RELIC_PIECES]
    powers = [(i, a, 0) for i, a in zip(field(ints, INT_AT, "player_power")[r].tolist(),
                                        field(floats, FLOAT_AT, "player_power_amount")[r].tolist()) if i > 0]
    mp = field(ints, INT_AT, "monster_power")[r].tolist()
    ma = field(floats, FLOAT_AT, "monster_power_amount")[r].tolist()
    for m in range(MAX_MONSTERS):
        for j in range(m * MAX_MONSTER_POWERS, (m + 1) * MAX_MONSTER_POWERS):
            if mp[j] > 0:
                powers.append((mp[j], ma[j], m + 1))
    cards = {}
    for k, p in enumerate(PILES):
        for i, u in zip(field(ints, INT_AT, p)[r].tolist(), field(floats, FLOAT_AT, p + "_upgrades")[r].tolist()):
            if i > 0:
                cards.setdefault((i, round(u)), [0, 0, 0])[k] += 1
    return relics, powers[:MAX_POWER_PIECES], sorted(cards.items())[:MAX_CARD_PIECES]


def check_pieces(ints, floats, rows):
    p = piece_ids(ints, floats, fixed=True)
    assert p["relic"].shape[1] == MAX_RELIC_PIECES and p["card"].shape[1] == MAX_CARD_PIECES
    for r in rows:
        relics, powers, cards = expected(ints, floats, r)
        n = len(relics)
        assert [(i, c) for i, c in zip(p["relic"][r, :n].tolist(), p["relic_counter"][r, :n].tolist())] == relics
        assert (p["relic"][r, n:] == 0).all()
        n = len(powers)
        got = list(zip(p["power"][r, :n].tolist(), p["power_amount"][r, :n].tolist(), p["power_owner"][r, :n].tolist()))
        assert got == powers, (r, got, powers)
        assert (p["power"][r, n:] == 0).all()
        n = len(cards)
        got = [((i, round(u)), c) for i, u, c in zip(p["card"][r, :n].tolist(), p["card_upgrades"][r, :n].tolist(),
                                                   p["card_counts"][r, :n].round().long().tolist())]
        assert got == cards, (r, got, cards)
        assert (p["card"][r, n:] == 0).all() and (p["card_counts"][r, n:] == 0).all()
    # Without `fixed`: as many as the batch's most, the same pieces.
    q = piece_ids(ints, floats)
    for name, x in q.items():
        assert torch.equal(x, p[name][:, :x.shape[1]]), name
        assert (p[name][:, x.shape[1]:] == 0).all(), name


def main():
    fights = positions(steps=60, practice=1.0, practice_decks="mix", character="defect")
    runs = positions(steps=150, practice=0.0, character="watcher")
    ints, floats, prac, legal = (torch.cat(x) for x in zip(fights, runs))
    combat = ints[:, RUN_INT_AT["screen"][0]] == COMBAT
    rows = torch.nonzero(combat).squeeze(1)
    p = piece_ids(ints[rows], floats[rows])
    assert (p["power_owner"] > 0).any() and (p["card_counts"] > 1).any() and (p["card_upgrades"] > 0).any() and (p["relic"] > 0).any(), \
        "the positions need monster powers, copies, upgrades and relics"
    check_pieces(ints[rows], floats[rows], range(len(rows)))
    print(f"piece_ids: ok ({len(rows)} fight positions; at most {p['relic'].shape[1]} relics, {p['power'].shape[1]} powers, "
          f"{p['card'].shape[1]} distinct pile cards)")

    # Past the caps: 30 relics, every power slot, 80 distinct cards in the draw pile plus copies elsewhere.
    big_i, big_f = ints[rows[:2]].clone(), floats[rows[:2]].clone()
    for name, n in (("relic", 30), ("player_power", None), ("monster_power", None), ("draw_pile", None)):
        a, b = INT_AT[name]
        n = n or b - a
        big_i[:, a:a + n] = torch.arange(1, n + 1) + 3
    a, b = FLOAT_AT["draw_pile_upgrades"]
    big_f[:, a:b] = (torch.arange(b - a) % 3 == 0).float()
    a, _ = INT_AT["discard_pile"]
    big_i[:, a:a + 5] = 4
    check_pieces(big_i, big_f, range(2))
    print("piece_ids at its caps: ok")

    torch.manual_seed(0)
    net = RunNet(d=32, layers=2, heads=2, unblended=True).eval()
    with torch.no_grad():
        want_logits, want_value = net(ints, floats, prac)

        # The order of the relics (in fights and between them) and of each pile's cards doesn't matter.
        shuffled_i, shuffled_f = ints.clone(), floats.clone()
        g = torch.Generator().manual_seed(1)
        for ids, nums, i_at, f_at in (("relic", "relic_counter", INT_AT, FLOAT_AT),
                                      ("run_relic", "run_relic_counter", RUN_INT_AT, RUN_FLOAT_AT)) + tuple(
                                          (p, p + "_upgrades", INT_AT, FLOAT_AT) for p in PILES):
            (a, b), (c, d) = i_at[ids], f_at[nums]
            for r in range(len(ints)):
                n = int((ints[r, a:b] > 0).sum())
                perm = torch.randperm(n, generator=g)
                shuffled_i[r, a:a + n] = ints[r, a:a + n][perm]
                shuffled_f[r, c:c + n] = floats[r, c:c + n][perm]
        assert not torch.equal(shuffled_i, ints)
        logits, value = net(shuffled_i, shuffled_f, prac)
        assert ((logits - want_logits).abs() * legal).max() < 1e-4 and (value - want_value).abs().max() < 1e-4
        print("the order of relics and pile cards: ok")

        # A different relic changes the output, in fights and between them.
        changed = ints.clone()
        for name, at in (("relic", INT_AT), ("run_relic", RUN_INT_AT)):
            a, _ = at[name]
            changed[:, a] = torch.where(changed[:, a] > 0, changed[:, a] % 50 + 1, changed[:, a])
        _, value = net(changed, floats, prac)
        has = (field(ints, INT_AT, "relic")[:, 0] > 0) & combat | (field(ints, RUN_INT_AT, "run_relic")[:, 0] > 0) & ~combat
        assert ((value - want_value).abs()[has] > 1e-6).float().mean() > 0.9

        # The fixed-shape pass, each kind of row, and FixedShapes over the whole mix.
        has_options = (field(ints, INT_AT, "option_card") > 0).any(1)
        for options in (False, True):
            r = torch.nonzero(combat & (has_options == options)).squeeze(1)
            logits, value = net.combat_static(ints[r], floats[r], prac[r], options)
            assert ((logits - want_logits[r]).abs() * legal[r]).max() < 1e-5, options
            assert (value - want_value[r]).abs().max() < 1e-5, options
        fixed = FixedShapes(net, compile=False)
        logits, value = fixed(ints, floats, prac)
        assert ((logits - want_logits).abs() * legal).max() < 1e-5 and (value - want_value).abs().max() < 1e-5
    print("combat_static and FixedShapes: ok")

    net.train()
    logits, value = net(ints, floats, prac)
    ((logits * legal).sum() + value.sum()).backward()
    params = dict(net.named_parameters())
    for name in NEW + ("relic.weight", "power.weight"):
        assert params[name].grad is not None and params[name].grad.abs().sum() > 0, name
    print("gradients to the new weights: ok")

    from training.train_runs import load
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "net.pt")
        torch.save({"net": net.state_dict(), "args": {"d": 32, "layers": 2, "heads": 2, "unblended": True}}, path)
        loaded, ck = load(path, "cpu")
        loaded.load_state_dict(ck["net"])
    assert loaded.unblended
    keys = set(RunNet(d=32, layers=2, heads=2).state_dict())
    assert not keys & set(NEW) and keys | set(NEW) == set(net.state_dict())
    print("checkpoints: ok")


if __name__ == "__main__":
    main()
