"""Checks for run_model.upgrade_state on an encoding 1 checkpoint (plain asserts, no pytest needed):

    python python/tests/test_upgrade.py

An encoding 1 checkpoint is made from a random encoding 2 network by taking its new weights out
(the hand's columns 8 and 9 become Feel No Pain's and Dark Embrace's again, with weights of their
own). Upgraded and loaded, the network keeps every old weight where it was, builds the same
network on every load, and plays as an encoding 1 network would: its outputs don't move when the
new inputs do (the stance, the last card type, the monsters' last moves, the new scalars, hand
columns and whole-play damage), in fights without orbs."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import numpy as np
import torch

from core.fight_model import FLOAT_AT, HAND_FEATS, INT_AT, MAX_HAND, MOVES_SEEN, SCALARS
from core.run_model import COMBAT, RUN_INT_AT, RunNet, _new_weights, upgrade_state
from core.sts_env import LAYOUT, RunEnv

V = LAYOUT["vocab"]


def encoding1_state(net):
    """`net`'s weights as an encoding 1 checkpoint would hold them."""
    s = {k: v.clone() for k, v in net.state_dict().items()}
    for k in _new_weights(net.d):
        del s[k]
    gen = torch.Generator().manual_seed(3)
    w = s["hand_feats.weight"]
    old = torch.empty(w.shape[0], 20)
    for block, at in ((0, 0), (1, HAND_FEATS)):
        old[:, block * 10:block * 10 + 8] = w[:, at:at + 8]
        old[:, block * 10 + 8:block * 10 + 10] = torch.randn(w.shape[0], 2, generator=gen)  # Feel No Pain, Dark Embrace
    s["hand_feats.weight"] = old
    w = s["scalars.weight"]
    s["scalars.weight"] = torch.cat([w[:, :16], w[:, SCALARS:SCALARS + 16]], dim=1)
    s["play.c.weight"] = torch.randn(w.shape[0], 4, generator=gen)
    s["kinds.weight"] = s["kinds.weight"][:8]
    return s


def combat_rows(character, n=32, steps=300, seed=5):
    env, rng = RunEnv(n, seed=seed, character=character), np.random.default_rng(seed)
    ints, floats = [], []
    for k in range(steps):
        obs = env.observe()
        if k % 10 == 9:
            fight = obs["ints"][:, RUN_INT_AT["screen"][0]] == COMBAT
            ints.append(obs["ints"][fight])
            floats.append(obs["floats"][fight])
        env.step([rng.choice(np.flatnonzero(m)) for m in obs["mask"]])
    ints, floats = np.concatenate(ints), np.concatenate(floats)
    no_orbs = (ints[:, INT_AT["orb"][0]:INT_AT["orb"][1]] == 0).all(1)
    return torch.from_numpy(ints[no_orbs]).long(), torch.from_numpy(floats[no_orbs])


def main():
    torch.manual_seed(0)
    fresh = RunNet(d=32, layers=2, heads=2)
    old = encoding1_state(fresh)
    kept = {k: v.clone() for k, v in old.items()}

    state = upgrade_state({k: v.clone() for k, v in old.items()})
    again = upgrade_state({k: v.clone() for k, v in old.items()})
    assert all(torch.equal(state[k], again[k]) for k in state), "two loads differ"
    net = RunNet(d=32, layers=2, heads=2)
    net.load_state_dict(state)  # strict: every weight is there
    net.eval()

    # The old weights stay where they were; the new ones that feed old tokens start at zero.
    s = net.state_dict()
    h = s["hand_feats.weight"]
    assert torch.equal(h[:, :8], kept["hand_feats.weight"][:, :8]) and torch.equal(h[:, HAND_FEATS:HAND_FEATS + 8], kept["hand_feats.weight"][:, 10:18])
    assert not h[:, 8:HAND_FEATS].any() and not h[:, HAND_FEATS + 8:].any()
    sc = s["scalars.weight"]
    assert torch.equal(sc[:, :16], kept["scalars.weight"][:, :16]) and torch.equal(sc[:, SCALARS:SCALARS + 16], kept["scalars.weight"][:, 16:])
    assert not sc[:, 16:SCALARS].any() and not sc[:, SCALARS + 16:].any()
    assert torch.equal(s["play.c.weight"][:, :4], kept["play.c.weight"]) and not s["play.c.weight"][:, 4:].any()
    assert torch.equal(s["kinds.weight"][:8], kept["kinds.weight"])
    for k in ("stance.weight", "card_type.weight", "moves_seen.weight"):
        assert not s[k].any(), k
    for k in ("move.weight", "orb.weight"):
        assert s[k].std() > 0, k

    # Its outputs don't depend on the new inputs.
    gen = np.random.default_rng(1)
    checked = 0
    for character in ("ironclad", "watcher"):
        ints, floats = combat_rows(character)
        with torch.no_grad():
            want = net(ints, floats)
        ints2, floats2 = ints.clone(), floats.clone()
        n = len(ints)
        ints2[:, INT_AT["stance"][0]] = torch.from_numpy(gen.integers(1, 5, n))
        ints2[:, INT_AT["last_played_type"][0]] = torch.from_numpy(gen.integers(0, 6, n))
        a, b = INT_AT["monster_moves"]
        ints2[:, a:b] = torch.from_numpy(gen.integers(0, V["move"], (n, b - a)))
        a = FLOAT_AT["scalars"][0]
        floats2[:, a + 16:a + SCALARS] = torch.from_numpy(gen.uniform(0, 5, (n, SCALARS - 16))).float()
        a = FLOAT_AT["hand"][0]
        hand = floats2[:, a:a + MAX_HAND * HAND_FEATS].reshape(n, MAX_HAND, HAND_FEATS)
        hand[:, :, 8:] = torch.from_numpy(gen.uniform(0, 3, (n, MAX_HAND, HAND_FEATS - 8))).float().round()
        a, b = FLOAT_AT["hand_target_total"]
        floats2[:, a:b] = torch.from_numpy(gen.uniform(0, 60, (n, b - a))).float()
        with torch.no_grad():
            got = net(ints2, floats2)
        assert torch.allclose(got[0], want[0], atol=1e-5) and torch.allclose(got[1], want[1], atol=1e-5), character
        # ...and the new inputs do reach a network trained on them (here: the fresh one).
        with torch.no_grad():
            assert not torch.allclose(fresh(ints2, floats2)[0], fresh(ints, floats)[0], atol=1e-5), character
        checked += n
    assert checked > 50, checked
    print(f"upgrade_state: ok ({checked} fight positions, moves embedding {V['move']} x {MOVES_SEEN})")


if __name__ == "__main__":
    main()
