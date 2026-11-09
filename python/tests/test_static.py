"""Checks for RunNet.combat_static, the fixed-shape combat pass, and core/fixed_shapes.py (plain
asserts, no pytest needed):

    python python/tests/test_static.py

On the Defect's practice-fight positions (orbs, and card selections now and then), the fixed-shape
pass gives the logits (of the legal moves: an option slot past a row's options is filled
differently, and masked out either way) and values of the usual one: rows without a selection
with no option tokens, rows with one with all 64; likewise RunNet.choices_static for choices between
fights, at every pair of caps the rows fit. FixedShapes (uncompiled here) gives the usual pass's
outputs on a batch of fights and choices between fights; `minibatches` puts rows in minibatches of
one kind, at most once, leaving out fewer than a minibatch of each fixed kind; and a PPO update runs
through them."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import contextlib
import types

import numpy as np
import torch

from core.fight_model import INT_AT
from core.fixed_shapes import CHOICE_CAPS, CHOICES, FIGHT, FIGHT_OPTIONS, FIGHT_ORBS, OTHER, FixedShapes, kinds, minibatch_size, minibatches
from core.ppo import ppo_update
from core.run_model import RunNet
from core.sts_env import RunEnv


def positions(steps=120, **kw):
    env, rng = RunEnv(32, seed=6, **kw), np.random.default_rng(6)
    ints, floats, prac, masks = [], [], [], []
    for _ in range(steps):
        obs = env.observe()
        ints.append(obs["ints"])
        floats.append(obs["floats"])
        prac.append(np.asarray(env.envs.practice_mask(), dtype=bool))
        masks.append(obs["mask"].astype(bool))
        env.step([rng.choice(np.flatnonzero(m)) for m in obs["mask"]])
    return (torch.from_numpy(np.concatenate(ints)).long(), torch.from_numpy(np.concatenate(floats)),
            torch.from_numpy(np.concatenate(prac)), torch.from_numpy(np.concatenate(masks)))


def main():
    ints, floats, prac, legal = positions(practice=1.0, practice_decks="mix", character="defect")
    has_options = (ints[:, INT_AT["option_card"][0]:INT_AT["option_card"][1]] > 0).any(1)
    assert has_options.any() and (~has_options).any(), "need both kinds of row"
    torch.manual_seed(0)
    net = RunNet(d=32, layers=2, heads=2).eval()
    with torch.no_grad():
        want_logits, want_value = net(ints, floats, prac)
        for options in (False, True):
            rows = torch.nonzero(has_options == options).squeeze(1)
            logits, value = net.combat_static(ints[rows], floats[rows], prac[rows], options)
            diff = ((logits - want_logits[rows]).abs() * legal[rows]).max()
            assert diff < 1e-5, (options, diff)
            assert torch.allclose(value, want_value[rows], atol=1e-5), options
    print(f"combat_static: ok ({int((~has_options).sum())} rows without options, {int(has_options.sum())} with)")

    # Without orb tokens, for fights with no orb channelled (the Watcher's, here), the same again.
    w_ints, w_floats, w_prac, w_legal = positions(practice=1.0, practice_decks="mix", character="watcher")
    no_orbs = (w_ints[:, INT_AT["orb"][0]:INT_AT["orb"][1]] == 0).all(1) & (w_ints[:, INT_AT["option_card"][0]:INT_AT["option_card"][1]] == 0).all(1)
    rows = torch.nonzero(no_orbs).squeeze(1)
    assert len(rows), "need fights without an orb"
    with torch.no_grad():
        want_logits, want_value = net(w_ints, w_floats, w_prac)
        logits, value = net.combat_static(w_ints[rows], w_floats[rows], w_prac[rows], False, orbs=False)
    assert ((logits - want_logits[rows]).abs() * w_legal[rows]).max() < 1e-5 and (value - want_value[rows]).abs().max() < 1e-5
    print(f"combat_static without orbs: ok ({len(rows)} rows)")

    # Fights and choices between fights together: half practice fights, half runs.
    mix = [positions(steps=60, practice=p, practice_decks="mix", character="defect") for p in (1.0, 0.0)]
    mix.append(positions(steps=60, practice=0.5, practice_decks="mix", character="watcher"))
    ints, floats, prac, legal = (torch.cat(x) for x in zip(*mix))
    k = kinds(ints)
    choice = torch.isin(k, torch.tensor(CHOICES))
    assert all((k == kind).any() for kind in (FIGHT, FIGHT_ORBS, FIGHT_OPTIONS)) and choice.any() and not (k == OTHER).any(), k.bincount()

    # The choices' fixed pass, at every pair of caps the rows fit (each row's own kind's caps, and
    # the larger ones).
    with torch.no_grad():
        want_logits, want_value = net(ints, floats, prac)
        for kind, (deck, opt) in CHOICE_CAPS.items():
            fits = [c for c, (d2, o2) in CHOICE_CAPS.items() if d2 <= deck and o2 <= opt]
            rows = torch.nonzero(torch.isin(k, torch.tensor(fits))).squeeze(1)
            logits, value = net.choices_static(ints[rows], floats[rows], deck, opt)
            assert ((logits - want_logits[rows]).abs() * legal[rows]).max() < 1e-5, (deck, opt)
            assert torch.allclose(value, want_value[rows], atol=1e-5), (deck, opt)
    print("choices_static: ok (" + ", ".join(f"caps {CHOICE_CAPS[c]}: {int((k == c).sum())} rows" for c in CHOICES if (k == c).any()) + ")")
    fixed = FixedShapes(net, compile=False)
    assert fixed.eval() is fixed and not net.training, "the rollout puts it in eval mode"
    with torch.no_grad():
        want_logits, want_value = net(ints, floats, prac)
        logits, value = fixed(ints, floats, prac)
    assert ((logits - want_logits).abs() * legal).max() < 1e-5 and (value - want_value).abs().max() < 1e-5
    some = torch.distributions.Categorical(logits=want_logits.masked_fill(~legal, -1e9)).sample()
    with torch.no_grad():
        got = fixed.logp(ints, floats, prac, legal, some, 64)  # chunks of 64 and 16, the last ones padded
        want = torch.log_softmax(want_logits.masked_fill(~legal, -1e9), -1).gather(1, some[:, None]).squeeze(1)
    assert (got - want).abs().max() < 1e-4, "the batch pass for --proximal"

    size = 256
    batches = minibatches(k, size)
    seen = torch.cat([rows for rows, _ in batches])
    assert len(seen.unique()) == len(seen), "no row twice"
    for kind in (FIGHT, FIGHT_ORBS, FIGHT_OPTIONS) + CHOICES:
        n = minibatch_size(kind, size)
        have, used = int((k == kind).sum()), int((k[seen] == kind).sum())
        if have < n:
            assert used == have, (kind, used, have)  # a kind too small for a minibatch: in the usual pass
        else:
            assert have - n < used <= have, (kind, used, have)  # fewer than a minibatch left over
    assert any(kind in CHOICES for _, kind in batches), "the choices' fixed minibatches"
    for rows, kind in batches:
        if kind == OTHER:
            assert len(rows) <= size
        else:
            assert len(rows) == minibatch_size(kind, size) and (k[rows] == kind).all(), (kind, len(rows))

    args = types.SimpleNamespace(epochs=2, minibatch=size, ent=0.01, clip=0.2, vf=0.5, max_grad=0.5, target_kl=1e9)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    before = [q.detach().clone() for q in net.parameters()]
    B = len(k)
    with torch.no_grad():
        lg, _ = net(ints, floats, prac)
        act = torch.distributions.Categorical(logits=lg.masked_fill(~legal, -1e9)).sample()
        logp = torch.log_softmax(lg.masked_fill(~legal, -1e9), -1).gather(1, act[:, None]).squeeze(1)
    stats, epochs = ppo_update(net, opt, args, contextlib.nullcontext, ints, floats, legal, act, logp, torch.randn(B), torch.randn(B),
                               practice=prac, batches=lambda: minibatches(k, size), forward=fixed.forward)
    assert epochs == 2 and all(np.isfinite(v) for v in stats.values()), stats
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters())), "the update changed nothing"
    print(f"fixed shapes: ok ({len(batches)} minibatches over {B} rows: " +
          f"fights {int((k == FIGHT).sum())}, with orbs {int((k == FIGHT_ORBS).sum())}, with a selection {int((k == FIGHT_OPTIONS).sum())}, "
          f"choices {int(choice.sum())})")


if __name__ == "__main__":
    main()
