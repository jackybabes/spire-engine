"""Checks for the next-move head (`RunNet(next_move=True)`, `train_runs.next_move_targets`), plain
asserts, no pytest needed:

    python python/tests/test_next_move.py

On random-play practice fights recorded as the trainer records them (`Envs.monster_next_moves`
beside each observation, and the episode ends), the labels are those a plain loop over the rollout
gives (the move the slot's monster shows at the next turn's first row of the same fight), and they
are nearly always the move the monster then makes: the encoding's last move for it two turns on.
Not always, for monsters that change their move during the player's turn (the Writhing Mass when
hit, the Guardian's mode shift, Lagavulin waking, Darklings coming back, Byrd knocked down): the
label is what the network will face at the next turn's start. The head learns the labels (on a
fixed network's features), plays no part (the same logits and values without it), and checkpoints
load with or without it; a release leaves it out."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from core.fight_model import INT_AT, MAX_MONSTERS
from core.run_model import COMBAT, MOVES_PER_MONSTER, RUN_INT_AT, RunNet
from core.sts_env import RunEnv
from training.train_runs import RS, TURN_AT, load, next_move_targets


def record(T=300, N=16):
    env, rng = RunEnv(N, seed=3, practice=1.0, practice_decks="mix", character="watcher"), np.random.default_rng(3)
    ints, floats, done, moves = [], [], [], []
    for _ in range(T):
        obs = env.observe()
        ints.append(obs["ints"])
        floats.append(obs["floats"])
        moves.append(np.asarray(env.envs.monster_next_moves(), dtype=np.int16).reshape(N, MAX_MONSTERS))
        status = env.step([rng.choice(np.flatnonzero(m)) for m in obs["mask"]])
        done.append(np.asarray(status) != 0)
    return (torch.from_numpy(np.stack(ints)), torch.from_numpy(np.stack(floats)), torch.from_numpy(np.stack(done)),
            torch.from_numpy(np.stack(moves)))


def main():
    ints, floats, done, moves = record()
    T, N = done.shape
    lab = next_move_targets(ints, floats, done, moves).reshape(T, N, MAX_MONSTERS)
    combat = ints[..., RUN_INT_AT["screen"][0]] == COMBAT
    mon = ints[..., INT_AT["monster"][0]:INT_AT["monster"][1]].long()
    seen = ints[..., INT_AT["monster_moves"][0]:INT_AT["monster_moves"][1]].long().reshape(T, N, MAX_MONSTERS, -1)
    turn = floats[..., TURN_AT]
    assert (lab[~combat] < 0).all()
    labelled = int((lab >= 0).sum())
    # The same labels from a plain loop: the next turn's first row in the same game, episode and floor.
    floor = floats[..., RS + 3]
    want = torch.full_like(lab, -1)
    for n in range(N):
        for t in range(T):
            if not combat[t, n]:
                continue
            for t2 in range(t + 1, T):
                if done[t2 - 1, n] or not combat[t2, n] or floor[t2, n] != floor[t, n] or turn[t2, n] > turn[t, n] + 1:
                    break
                if turn[t2, n] == turn[t, n] + 1:
                    for j in range(MAX_MONSTERS):
                        if mon[t, n, j] > 0 and mon[t2, n, j] == mon[t, n, j] and moves[t2, n, j] > 0:
                            want[t, n, j] = (int(moves[t2, n, j]) - 1) % MOVES_PER_MONSTER
                    break
    assert torch.equal(lab, want)
    # Each label against the monster's last move two turns on, where the rollout gets that far in the
    # same fight with the same monster in the slot.
    checked = agree = 0
    for t, n, j in torch.nonzero(lab >= 0).tolist():
        for t2 in range(t + 1, T):
            if done[t2 - 1, n] or not combat[t2, n] or turn[t2, n] > turn[t, n] + 2:
                break
            if turn[t2, n] == turn[t, n] + 2:
                if mon[t2, n, j] == mon[t, n, j]:
                    checked += 1
                    want = 1 + (int(mon[t, n, j]) - 1) * MOVES_PER_MONSTER + int(lab[t, n, j])
                    agree += int(seen[t2, n, j, 0]) == want
                break
    assert labelled > 500 and checked > 200, (labelled, checked)
    assert agree > 0.95 * checked, (agree, checked)
    rows = int(((lab >= 0).any(-1)).sum())
    print(f"labels: ok ({labelled} monster labels on {rows} of {int(combat.sum())} fight rows, as a plain loop gives them; "
          f"{agree} of {checked} checked against the move then made agree)")

    # The head learns them, on a fixed network's features.
    flat_i, flat_f = ints.reshape(T * N, -1).long(), floats.reshape(T * N, -1)
    lab = lab.reshape(T * N, -1)
    has_rows = torch.nonzero((lab >= 0).any(1)).squeeze(1)
    torch.manual_seed(0)
    net = RunNet(d=32, layers=2, heads=2, next_move=True)
    opt = torch.optim.Adam(net.move_head.parameters(), lr=1e-2)

    def loss_on(rows):
        pred = net.next_move_pred(flat_i[rows], flat_f[rows])
        lb = lab[rows]
        return F.cross_entropy(pred[lb >= 0], lb[lb >= 0])
    with torch.no_grad():
        before = float(loss_on(has_rows))
    g = torch.Generator().manual_seed(0)
    for _ in range(150):
        loss = loss_on(has_rows[torch.randint(len(has_rows), (256,), generator=g)])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        after = float(loss_on(has_rows))
    assert after < 0.7 * before, (before, after)
    print(f"the head learns: loss {before:.3f} -> {after:.3f}")

    # The head plays no part: the same weights without it give the same logits and values.
    plain = RunNet(d=32, layers=2, heads=2)
    missing, unexpected = plain.load_state_dict(net.state_dict(), strict=False)
    assert not missing and all(k.startswith("move_head.") for k in unexpected)
    some = torch.arange(0, T * N, 7)
    with torch.no_grad():
        l1, v1 = net(flat_i[some], flat_f[some])
        l2, v2 = plain(flat_i[some], flat_f[some])
    assert torch.equal(l1, l2) and torch.equal(v1, v2)
    print("the head changes no play: ok")

    from release.export_release import export, load_release
    with tempfile.TemporaryDirectory() as d:
        args = {"d": 32, "layers": 2, "heads": 2, "move_head": True}
        torch.save({"net": net.state_dict(), "args": args}, os.path.join(d, "with.pt"))
        torch.save({"net": plain.state_dict(), "args": {**args, "move_head": False}}, os.path.join(d, "without.pt"))
        got, ck = load(os.path.join(d, "with.pt"), "cpu")
        got.load_state_dict(ck["net"])
        assert got.move_head is not None
        got, ck = load(os.path.join(d, "without.pt"), "cpu")
        got.load_state_dict(ck["net"])
        assert got.move_head is None
        meta = export(net, {"args": args}, os.path.join(d, "net.safetensors"))
        released, _ = load_release(os.path.join(d, "net.safetensors"))
        # (the file's parameters: a release leaves out the deck lookup's inputs, all zero)
    assert released.move_head is None and int(meta["parameters"]) == sum(p.numel() for n, p in released.named_parameters()
                                                                     if not n.startswith(("lookup_opt.", "lookup_run.", "lookup_pick.")))
    print("checkpoints and releases: ok")


if __name__ == "__main__":
    main()
