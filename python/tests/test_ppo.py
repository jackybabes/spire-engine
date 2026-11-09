"""Checks for core.ppo's early stop (plain asserts, no pytest needed):

    python python/tests/test_ppo.py

With minibatches of one kind each (core.fixed_shapes), a small kind whose rows moved far (their
recorded log-probabilities well off) no longer stops the update by itself: the stop watches the
whole policy's KL, each kind weighted by its rows. Every row off still stops it at once, and
mixed minibatches stop as before (the latest minibatch's KL). The per-kind KL and the stop are
reported. Anchored on the network at the update's start (`logp_prox`), a rollout played by an
older policy doesn't stop the update. Value-only epochs (`policy=False`, train_runs.py
--value-epochs) lower the value loss, and on a network whose policy and value share weights the
`clone` penalty keeps the policy much nearer where it was than without it; the gradient's norm
before clipping is reported. The zero-avoiding term (`zero_avoid`, the KL from the uniform policy
over the legal moves) brings back a move the policy has given up on far more than an entropy
bonus of the same weight, leaves a row with one legal move alone, and is reported (`kl_uniform`)."""
import argparse
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import torch
import torch.nn.functional as F

from core.ppo import ppo_update

B, A = 4096 + 128, 6


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, A + 1)

    def forward(self, ints, floats, practice=None):
        out = self.lin(floats)
        return out[:, :A], out[:, A]


class SharedToy(torch.nn.Module):
    """The policy and the value from one hidden layer, as the run network's heads share its encoder."""

    def __init__(self):
        super().__init__()
        self.body = torch.nn.Linear(8, 16)
        self.lin = torch.nn.Linear(16, A + 1)

    def forward(self, ints, floats, practice=None):
        out = self.lin(torch.tanh(self.body(floats)))
        return out[:, :A], out[:, A]


def run(off_rows, batches=None, epochs=2, proximal=False, aux=None):
    torch.manual_seed(0)
    net = Toy()
    if aux:
        aux = aux(net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    args = argparse.Namespace(epochs=epochs, minibatch=1024, clip=0.2, vf=0.5, ent=0.0, max_grad=0.5, target_kl=0.05, prox_cap=2.0)
    ints, floats = torch.zeros(B, 1, dtype=torch.long), torch.randn(B, 8)
    mask = torch.ones(B, A, dtype=torch.bool)
    act = torch.randint(0, A, (B,))
    with torch.no_grad():
        logp = F.log_softmax(net(ints, floats)[0], -1).gather(1, act[:, None]).squeeze(1)
    prox = logp.clone()  # the network at the update's start
    logp[off_rows] -= 1.0  # these rows' policy moved far since the rollout
    forward = (lambda kind, i, f, p: net(i, f, p)) if batches else None
    return ppo_update(net, opt, args, contextlib.nullcontext, ints, floats, mask, act, logp, torch.randn(B), torch.randn(B),
                      practice=torch.ones(B, dtype=torch.bool), batches=batches, forward=forward, logp_prox=prox if proximal else None, aux=aux)


def main():
    small = torch.arange(4096, B)  # 128 rows of kind 1 (3%), the rest kind 0
    def one_kind_first():
        return [(small, 1)] + [(torch.arange(i, i + 1024), 0) for i in range(0, 4096, 1024)]

    stats, epochs = run(small, one_kind_first)
    assert epochs == 2 and "stopped_at" not in stats, (epochs, stats)
    assert stats["approx_kl_1"] > 0.05 > stats["approx_kl_0"], stats

    stats, epochs = run(torch.arange(B), one_kind_first)
    assert epochs == 1 and stats["stopped_at"] == 2 / 5 and stats["stop_kind"] == 0, (epochs, stats)

    stats, epochs = run(torch.arange(B))
    assert epochs == 1 and stats["stopped_at"] == 1 / 5 and stats["stop_kind"] is None, (epochs, stats)
    stats, epochs = run(small)
    assert epochs == 2 and "stopped_at" not in stats, (epochs, stats)
    stats, epochs = run(torch.arange(B), proximal=True)
    assert epochs == 2 and "stopped_at" not in stats and stats["approx_kl"] < 0.01, (epochs, stats)
    print("ppo early stop: ok")

    # An aux loss that skips minibatches (train_runs --human-every) returns a loss with no gradient
    # there: no backward for it, and its stats come only from the minibatches it ran on.
    def every_other(net):
        calls = [0]

        def aux():
            calls[0] += 1
            if calls[0] % 2:
                return torch.zeros(()), {}
            return net.lin.weight.square().sum() * 1e-3, {"aux_ran": 1.0}
        return aux
    stats, epochs = run(small, aux=every_other)
    assert epochs == 2 and stats["aux_ran"] == 1.0, (epochs, stats)
    print("aux skipping minibatches: ok")

    # A minibatch of one row (a kind's leftover) leaves the weights finite.
    def with_single():
        return [(torch.arange(0, 1), 1)] + [(torch.arange(i, i + 1024), 0) for i in range(1, 4097, 1024)]
    torch.manual_seed(0)
    net = Toy()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    args = argparse.Namespace(epochs=1, minibatch=1024, clip=0.2, vf=0.5, ent=0.0, max_grad=0.5, target_kl=0.05, prox_cap=2.0)
    ints, floats = torch.zeros(B, 1, dtype=torch.long), torch.randn(B, 8)
    act = torch.randint(0, A, (B,))
    with torch.no_grad():
        logp = F.log_softmax(net(ints, floats)[0], -1).gather(1, act[:, None]).squeeze(1)
    ppo_update(net, opt, args, contextlib.nullcontext, ints, floats, torch.ones(B, A, dtype=torch.bool), act, logp, torch.randn(B), torch.randn(B),
               practice=torch.ones(B, dtype=torch.bool), batches=with_single, forward=lambda kind, i, f, p: net(i, f, p))
    assert all(torch.isfinite(q).all() for q in net.parameters())
    print("one-row minibatch: ok")

    # Value-only epochs: SharedToy's policy and value share a layer, so training the value moves
    # the policy unless the clone penalty holds it.
    def value_phase(clone):
        torch.manual_seed(0)
        net = SharedToy()
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        args = argparse.Namespace(epochs=8, minibatch=1024, clip=0.2, vf=0.5, ent=0.0, max_grad=0.5, target_kl=10.0, prox_cap=2.0)
        ints, floats = torch.zeros(B, 1, dtype=torch.long), torch.randn(B, 8)
        mask = torch.ones(B, A, dtype=torch.bool)
        act = torch.randint(0, A, (B,))
        ret = floats @ torch.linspace(-1, 1, 8) + 2.0
        with torch.no_grad():
            start = F.log_softmax(net(ints, floats)[0], -1)
            v0 = 0.5 * (net(ints, floats)[1] - ret).pow(2).mean()
        keep = start.gather(1, act[:, None]).squeeze(1)
        stats, epochs = ppo_update(net, opt, args, contextlib.nullcontext, ints, floats, mask, act, keep, torch.randn(B), ret,
                                   policy=False, practice=torch.ones(B, dtype=torch.bool), logp_prox=keep, clone=clone)
        with torch.no_grad():
            now = F.log_softmax(net(ints, floats)[0], -1)
            v1 = 0.5 * (net(ints, floats)[1] - ret).pow(2).mean()
        drift = (start.exp() * (start - now)).sum(-1).mean()  # KL(start || now) over every move
        assert epochs == 8 and stats["grad_norm"] > 0 and 0 <= stats["grad_clipped"] <= 1, stats
        return float(v0), float(v1), float(drift)
    v0, v1, free = value_phase(0.0)
    w0, w1, held = value_phase(10.0)
    assert v1 < 0.5 * v0 and w1 < 0.5 * w0, (v0, v1, w0, w1)
    assert held < 0.25 * free, (held, free)
    print(f"value-only epochs: ok (value loss {v0:.3f} -> {v1:.3f} free, -> {w1:.3f} held; policy KL {free:.4f} free, {held:.4f} held)")


    # The zero-avoiding term: a move at about 1e-4 (its bias -9), no advantage to follow; the KL
    # from uniform brings it back, the entropy bonus at the same weight hardly does. Plain SGD so the
    # gradients' sizes show (in the network they share weights with the policy gradient, which sets
    # Adam's scale; alone on a parameter Adam would step either term at full size), the value off.
    def revive(ent, za):
        torch.manual_seed(0)
        net = Toy()
        with torch.no_grad():
            net.lin.bias[0] = -9.0
        opt = torch.optim.SGD(net.parameters(), lr=10.0)
        args = argparse.Namespace(epochs=8, minibatch=1024, clip=0.2, vf=0.0, ent=ent, max_grad=0.5, target_kl=10.0, prox_cap=2.0)
        ints, floats = torch.zeros(B, 1, dtype=torch.long), torch.randn(B, 8)
        mask = torch.ones(B, A, dtype=torch.bool)
        mask[:64, 1:] = False  # rows with one legal move: no KL, no NaN
        act = torch.randint(1, A, (B,))
        act[:64] = 0
        with torch.no_grad():
            logp = F.log_softmax(net(ints, floats)[0].masked_fill(~mask, -1e9), -1).gather(1, act[:, None]).squeeze(1)
        stats, _ = ppo_update(net, opt, args, contextlib.nullcontext, ints, floats, mask, act, logp, torch.zeros(B), torch.zeros(B),
                              practice=torch.ones(B, dtype=torch.bool), zero_avoid=torch.full((B,), za) if za else None)
        assert all(torch.isfinite(q).all() for q in net.parameters())
        with torch.no_grad():
            p0 = F.softmax(net(ints, floats)[0][64:], -1)[:, 0].mean()
        return float(p0), stats["kl_uniform"]
    base, kl0 = revive(0.0, 0.0)
    ent, _ = revive(0.05, 0.0)
    za, kl1 = revive(0.0, 0.05)
    assert za > 3 * base and ent < 1.2 * base, (base, ent, za)
    assert kl1 < kl0, (kl0, kl1)
    print(f"zero-avoiding term: ok (the given-up move at {base:.1e} alone, {ent:.1e} with entropy 0.05, {za:.1e} with zero-avoid 0.05)")

if __name__ == "__main__":
    main()
