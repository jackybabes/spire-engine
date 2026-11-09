"""Checks for the fight-outcome head (plain asserts, no pytest needed), on the CPU with a tiny untrained network:

    python python/tests/test_fight_outcome.py

RunNet(fight_outcome=True) predicts (B, 2) for fight rows, and its loss trains the head alone (the network reads
detached: no other parameter gets a gradient); a rollout with --fight-outcome labels each fight row whose practice
fight ended in it with that fight's ending (won 0/1, HP share left, 0 when lost), from the fight's first row in the
rollout to its last; and the deck network's lookup (core/lookup.py) scores a human card reward's every option against
this act's boss and the next act's bosses, skipping first, with no gain over itself."""
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import torch

import json

import numpy as np

from core.lookup import Lookup
from core.run_model import RunNet, fight_outcome_loss
from training.train_runs import Collector, freeze_others, head_update, optimizer_groups, parse_args, rollout_buffers


def main():
    args = parse_args(["--name", "test", "--character", "defect", "--practice", "1", "--practice-decks", "mix", "--envs", "16",
                       "--steps", "48", "--eval-runs", "0", "--fight-outcome", "0.5"])
    torch.manual_seed(0)
    net = RunNet(d=16, layers=1, heads=2, fight_outcome=True)
    col = Collector(args, torch.device("cpu"), contextlib.nullcontext, 0)
    out = rollout_buffers(args.steps, args.envs, "cpu")
    ro = col.collect(0, net, out)
    has, won, hp = out["fo_has"], out["fo_won"], out["fo_hp"]
    assert has.any() and ro["practice_now"], "some fights ended and labelled their rows"
    assert set(won[has].unique().tolist()) <= {0.0, 1.0} and ((hp >= 0) & (hp <= 1)).all()
    assert (hp[has & (won == 0)] == 0).all(), "a lost fight leaves no HP"
    assert (won[~has] == 0).all() and (hp[~has] == 0).all()
    # A game's labelled rows run from the rollout's start to its last fight's end: nothing labelled after an unlabelled row.
    for g in range(args.envs):
        col_has = has[:, g].tolist()
        last = max((t for t, h in enumerate(col_has) if h), default=-1)
        assert all(col_has[:last + 1]), (g, col_has)
    ended = sum(ro["practice_now"][i]["won"] for i in range(len(ro["practice_now"])))
    assert ended >= 0

    rows = torch.nonzero(has.reshape(-1)).squeeze(1)[:32]
    ints, floats = out["ints"].reshape(-1, out["ints"].shape[-1])[rows].long(), out["floats"].reshape(-1, out["floats"].shape[-1])[rows]
    pred = net.fight_outcome_pred(ints, floats)
    assert pred.shape == (len(rows), 2)
    loss, st = fight_outcome_loss(pred, won.reshape(-1)[rows], hp.reshape(-1)[rows])
    loss.backward()
    head = {n for n, _ in net.named_parameters() if n.startswith("fight_outcome_head.")}
    assert all(p.grad is not None for n, p in net.named_parameters() if n in head), "the head learns"
    assert all(p.grad is None for n, p in net.named_parameters() if n not in head), "nothing else gets a gradient"
    assert 0 <= float(st["fight_outcome_acc"]) <= 1 and 0 <= float(st["fight_outcome_brier"]) <= 1
    # --head-only's update: optimizer steps on the head alone; every other weight exactly as it was.
    net.zero_grad(set_to_none=True)
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    opt = torch.optim.Adam(net.fight_outcome_head.parameters(), lr=1e-2)

    def step():
        l2, s2 = fight_outcome_loss(net.fight_outcome_pred(ints, floats), won.reshape(-1)[rows], hp.reshape(-1)[rows])
        return l2, {k: float(v) for k, v in s2.items()}
    hst = head_update(step, opt, 3)
    assert hst["approx_kl"] == 0 and "fight_outcome_acc" in hst
    assert all(torch.equal(p, before[n]) for n, p in net.named_parameters() if n not in head), "the policy and value untouched"
    assert any(not torch.equal(p, before[n]) for n, p in net.named_parameters() if n in head), "the head moved"
    print(f"fight outcome: ok ({int(has.sum())} of {has.numel()} rows labelled, {len(ro['practice_now'])} fights ended, "
          f"{ended} won; loss {loss.item():.3f}, the head alone learns)")

    rec = {"play_id": "x", "ascension": 20, "won": True, "floor": 5, "act": 1, "hp": 60, "max_hp": 80, "gold": 99, "room": "M",
           "boss": "Hexaghost", "ruby_key": False, "relics": ["Burning Blood"], "kind": "card",
           "deck": ["Strike_R"] * 5 + ["Defend_R"] * 4 + ["Bash"], "offered": ["Inflame", "Shrug It Off", "Clothesline+1"], "taken": "Inflame"}
    lk = Lookup(net.eval(), torch.device("cpu"), contextlib.nullcontext, draws=2)
    sc = lk.human_cards([json.dumps(rec), json.dumps({**rec, "kind": "campfire", "taken": "REST"})])
    assert list(sc) == [0], "the card decision only"
    s0 = sc[0]
    assert s0.options == [-1, 0, 1, 2] and s0.win.shape == (4, 4), (s0.options, s0.win.shape)  # Act 1: the boss and Act 2's three
    assert ((s0.win >= 0) & (s0.win <= 1)).all()
    dw, _ = s0.gain()
    assert np.allclose(dw[0], 0), "skipping gains nothing over itself"
    print(f"lookup: ok (a card reward's 4 options x 4 probe fights; gains against this act's boss {np.round(dw[1:, 0], 3).tolist()})")

    # Lookup.fill: the games at a card reward get their options' gains and the run's own scores; the others none. A
    # network's new inputs start at zero, so the filled lookup changes nothing until it learns to use them.
    from core.lookup import CARD_REWARD, SCREEN_AT
    from core.sts_env import LAYOUT
    from core.run_model import RUN_FLOAT_AT
    from core.sts_env import RunEnv
    env, rng = RunEnv(32, seed=3, character="ironclad", ascension=20), np.random.default_rng(3)
    for _ in range(4000):
        obs = env.observe()
        if (obs["ints"][:, SCREEN_AT] == CARD_REWARD).sum() >= 2:
            break
        env.step([rng.choice(np.flatnonzero(m)) for m in obs["mask"]])
    at = obs["ints"][:, SCREEN_AT] == CARD_REWARD
    assert at.sum() >= 2, "random runs reach card rewards"
    plain = {k: np.array(v) for k, v in obs.items()}
    lk.fill(env.envs, obs)
    (oa, ob), (ra, rb) = RUN_FLOAT_AT["opt_lookup"], RUN_FLOAT_AT["run_lookup"]
    assert (obs["floats"][~at][:, oa:rb] == 0).all() and (obs["floats"][~at] == plain["floats"][~at]).all(), "others untouched"
    assert (obs["floats"][at][:, ra:rb] != 0).any(1).all(), "the run's own scores"
    assert (obs["floats"][at][:, oa:ob] != 0).any(1).all(), "the cards' gains"
    outside = np.ones(obs["floats"].shape[1], bool)
    outside[oa:rb] = False
    assert (obs["floats"][:, outside] == plain["floats"][:, outside]).all(), "nothing else changes"
    rows = torch.from_numpy(np.flatnonzero(at))
    with torch.no_grad():
        a_l, a_v = net(torch.from_numpy(plain["ints"][rows]).long(), torch.from_numpy(plain["floats"][rows]))
        b_l, b_v = net(torch.from_numpy(obs["ints"][rows]).long(), torch.from_numpy(obs["floats"][rows]))
    assert torch.equal(a_l, b_l) and torch.equal(a_v, b_v), "zero-initialised inputs change nothing"
    with torch.no_grad():
        net.lookup_opt.weight.fill_(0.1)
        c_l, _ = net(torch.from_numpy(obs["ints"][rows]).long(), torch.from_numpy(obs["floats"][rows]))
    assert not torch.equal(b_l, c_l), "a trained lookup input reaches the choice"
    # The lookup's pick weights: each option's pick logit moves by the screen's weights times its four numbers, nothing else.
    from core.run_model import PICK
    with torch.no_grad():
        net.lookup_opt.weight.zero_()
        d_l, _ = net(torch.from_numpy(obs["ints"][rows]).long(), torch.from_numpy(obs["floats"][rows]))
        net.lookup_pick.weight[CARD_REWARD] = torch.tensor([2.0, 0.0, 1.0, 0.0])
        e_l, _ = net(torch.from_numpy(obs["ints"][rows]).long(), torch.from_numpy(obs["floats"][rows]))
        net.lookup_pick.weight.zero_()
    o_feats = torch.from_numpy(obs["floats"][rows][:, oa:ob]).reshape(len(rows), -1, 4)
    want = 2 * o_feats[:, :, 0] + o_feats[:, :, 2]
    n_opt = (e_l - d_l).shape[1] - PICK
    diff = e_l - d_l
    assert torch.allclose(diff[:, PICK:PICK + want.shape[1]][:, :n_opt], want[:, :n_opt], atol=1e-4)
    diff[:, PICK:PICK + want.shape[1]] = 0
    assert diff.abs().max() < 1e-4, "only the pick logits move"
    print(f"lookup fill: ok ({int(at.sum())} games at a card reward filled, the rest untouched)")

    # The fixed-shape head (FixedShapes.fight_outcome, uncompiled here) gives fight_outcome_pred's outputs, on the
    # rollout's fight rows; and the lookup scores the same with it, its probes in reused buffers.
    from core.fixed_shapes import FixedShapes
    torch.manual_seed(1)
    net2 = RunNet(d=16, layers=1, heads=2, fight_outcome=True).eval()
    fixed = FixedShapes(net2, compile=False)
    flat_i, flat_f = out["ints"].reshape(-1, out["ints"].shape[-1]).long(), out["floats"].reshape(-1, out["floats"].shape[-1])
    combat = torch.nonzero(flat_i[:, SCREEN_AT] == 3).squeeze(1)[:300]  # encode_run.rs screen_id: combat
    with torch.no_grad():
        want = net2.fight_outcome_pred(flat_i[combat], flat_f[combat])
        got = fixed.fight_outcome(flat_i[combat], flat_f[combat])
    assert torch.allclose(want, got, atol=1e-5), (want - got).abs().max()
    games = np.flatnonzero(at)
    plain_lk = Lookup(net2, torch.device("cpu"), contextlib.nullcontext, draws=2)
    fixed_lk = Lookup(net2, torch.device("cpu"), contextlib.nullcontext, draws=2, fixed=fixed)
    a_sc, b_sc = plain_lk.card_rewards(env.envs, games), fixed_lk.card_rewards(env.envs, games)
    assert set(a_sc) == set(b_sc) == set(games.tolist())
    for g in a_sc:
        assert a_sc[g].options == b_sc[g].options and np.allclose(a_sc[g].win, b_sc[g].win, atol=1e-5), g
    # Probes written as only the head's columns (keep) read the same as full rows.
    from core.lookup import KEEP_FLOATS, KEEP_INTS
    meta, ib, fb = env.envs.probe_card_rewards(games.tolist(), 2, 1)
    full = plain_lk._head(*plain_lk._rows(ib, fb, len(meta)))
    bi, bf = bytearray(len(meta) * KEEP_INTS * 4), bytearray(len(meta) * KEEP_FLOATS * 4)
    meta2, n2 = env.envs.probe_card_rewards_into(games.tolist(), 2, 1, bi, bf, (KEEP_INTS, KEEP_FLOATS))
    part = plain_lk._head(*plain_lk._rows(bi, bf, n2, (KEEP_INTS, KEEP_FLOATS)))
    assert meta == meta2 and np.allclose(full, part, atol=1e-6), np.abs(full - part).max()
    print(f"lookup with fixed shapes: ok ({len(combat)} fight rows; {len(games)} card rewards scored the same; "
          f"{len(meta)} probes the same from {KEEP_INTS} ints and {KEEP_FLOATS} floats a row)")

    # Lookup(every=True): card rewards score as card_rewards does; rest sites, shops and the rest get their options' gains
    # in their slots and the run's own scores; screens it doesn't probe (fights, the map, events) nothing.
    from core.lookup import PROBED
    every = Lookup(net2, torch.device("cpu"), contextlib.nullcontext, draws=2, fixed=fixed, every=True)
    c_sc = every.options(env.envs, games)
    for g in a_sc:
        assert c_sc[g].options == a_sc[g].options and np.allclose(c_sc[g].win, a_sc[g].win, atol=1e-5), g
    env2, rng2 = RunEnv(64, seed=7, character="ironclad", ascension=20), np.random.default_rng(7)
    seen = set()
    for _ in range(3000):
        obs = env2.observe()
        screens = obs["ints"][:, SCREEN_AT]
        at = np.isin(screens, PROBED) & (screens != CARD_REWARD)
        if at.any():
            plain = np.array(obs["floats"])
            every.fill(env2.envs, obs)
            filled = (obs["floats"][:, ra:rb] != 0).any(1)
            assert not filled[~np.isin(screens, PROBED)].any(), "only probed screens"
            assert (obs["floats"][:, outside] == plain[:, outside]).all(), "nothing else changes"
            for g in np.flatnonzero(at & filled):
                seen.add(int(screens[g]))
        if len(seen) >= 3:
            break
        env2.step([rng2.choice(np.flatnonzero(m)) for m in obs["mask"]])
    names = {i + 1: k for i, k in enumerate(LAYOUT["screens"])}
    assert len(seen) >= 3, [names[i] for i in seen]
    # fill writes all the games at once (Lookup._feats): what write() writes game by game from the same scores.
    from core.lookup import write
    obs = env2.observe()
    probed = np.flatnonzero(np.isin(obs["ints"][:, SCREEN_AT], PROBED))
    one = {k: np.array(v) for k, v in obs.items()}
    every.fill(env2.envs, obs)
    for g, sc in every.options(env2.envs, probed).items():
        write(one["floats"][g], sc)
    from core.lookup import MAP
    at_map = obs["ints"][:, SCREEN_AT] == MAP
    assert np.allclose(obs["floats"][~at_map], one["floats"][~at_map], atol=1e-5), np.abs(obs["floats"] - one["floats"]).max()
    # The map: paths into elites and hallway fights get a win chance and an HP share (not gains), the run token nothing.
    m_opt = obs["floats"][at_map][:, oa:ob].reshape(int(at_map.sum()), -1, 4)
    assert at_map.any() and (m_opt[:, :, :2] > 0).any(), "some path scored"
    assert ((m_opt[:, :, :2] >= 0) & (m_opt[:, :, :2] <= 1)).all() and (m_opt[:, :, 2:] == 0).all()
    assert (obs["floats"][at_map][:, ra:rb] == 0).all()
    # Lookup.last: the filled games' best option slots (-1 nothing), which a rollout counts against the moves taken.
    from core.run_model import PICK
    from training.train_runs import RolloutCounts
    src, best, nothing, scr = every.last
    assert set(src.tolist()) <= set(probed.tolist()) and len(best) == len(src) and (scr == obs["ints"][src, SCREEN_AT]).all()
    assert (best.any(1) != nothing).all(), "a best option, or nothing"
    acc = RolloutCounts()
    acts = np.zeros(len(obs["ints"]), dtype=np.int64)
    acts[src] = np.where(nothing, 0, PICK + best.argmax(1))
    acc.count_lookup(acts, src, best, nothing, scr)
    agree = acc.summary()["lookup_agree"]
    assert all(v == 1.0 for v in agree.values() if v != ""), agree
    # A whole rollout with the lookup at every choice: its time and the agreement come back in the summary.
    args2 = parse_args(["--name", "test", "--character", "ironclad", "--ascension", "20", "--envs", "16", "--steps", "64",
                        "--eval-runs", "0"])
    col2 = Collector(args2, torch.device("cpu"), contextlib.nullcontext, 0)
    col2.lookup = Lookup(net2, torch.device("cpu"), contextlib.nullcontext, draws=2, every=True)
    ro2 = col2.collect(0, RunNet(d=16, layers=1, heads=2).eval(), rollout_buffers(args2.steps, args2.envs, "cpu"))
    assert ro2["lookup"]["rows"] > 0 and any(v != "" for v in ro2["lookup_agree"].values()), (ro2["lookup"], ro2["lookup_agree"])
    # --lookup-lr-mult: the lookup's five tensors in a group of their own at that multiple; every parameter in one group.
    g2 = optimizer_groups(net2, parse_args(["--name", "t", "--lookup-lr-mult", "10"]))
    assert len(g2) == 2 and g2[1]["lr_mult"] == 10 and len(g2[1]["params"]) == 5  # (lookup_opt, lookup_run, lookup_pick)
    grouped = [id(q) for g in g2 for q in g["params"]]
    assert sorted(grouped) == sorted(id(q) for q in net2.parameters())
    assert len(optimizer_groups(net2, parse_args(["--name", "t"]))) == 1
    g3 = optimizer_groups(net2, parse_args(["--name", "t", "--lookup-lr-mult", "10", "--lookup-pick-lr-mult", "100"]))
    assert [len(g["params"]) for g in g3][1:] == [4, 1] and [g.get("lr_mult") for g in g3] == [None, 10, 100]
    assert sorted(id(q) for g in g3 for q in g["params"]) == sorted(id(q) for q in net2.parameters())
    # --train-only: just the named parameters, in their groups at their multiples.
    g4 = optimizer_groups(net2, parse_args(["--name", "t", "--lookup-lr-mult", "10", "--lookup-pick-lr-mult", "100",
                                            "--train-only", "lookup_,run_value"]))
    param_names = {id(q): n for n, q in net2.named_parameters()}
    kept = [param_names[id(q)] for g in g4 for q in g["params"]]
    assert kept and all(n.startswith(("lookup_", "run_value")) for n in kept), kept
    assert len(kept) == sum(1 for n, _ in net2.named_parameters() if n.startswith(("lookup_", "run_value")))
    assert [g.get("lr_mult") for g in g4] == [None, 10, 100]
    # ... and the rest take no gradients: a backward pass leaves them without one, the named ones with one.
    net5 = RunNet(d=16, layers=1, heads=2, fight_outcome=True)
    opt5 = torch.optim.Adam(optimizer_groups(net5, parse_args(["--name", "t", "--train-only", "lookup_,run_value"])), lr=1e-3)
    freeze_others(net5, opt5)
    sub = torch.from_numpy(np.flatnonzero(at))
    lg5, v5 = net5(torch.from_numpy(obs["ints"][sub]).long(), torch.from_numpy(obs["floats"][sub]))
    (lg5.float().sum() + v5.sum()).backward()
    for n, q in net5.named_parameters():
        assert (q.grad is not None) == n.startswith(("lookup_", "run_value")), n
    print(f"lookup in a rollout: ok ({int(ro2['lookup']['rows'])} probe rows; agreement {ro2['lookup_agree']})")
    print(f"lookup at every choice: ok (card rewards as before; filled at {sorted(names[i] for i in seen)})")


if __name__ == "__main__":
    main()
