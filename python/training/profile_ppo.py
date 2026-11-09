"""Where PPO's time goes, on a GPU: a training step's rows a second by kind of row
(`core.fixed_shapes`: fights with and without a selection, choices between fights by their caps),
what that comes to for an update of 262,144 positions and 2 epochs, and an op profile of the fight
pass uncompiled (CUDA graphs hide their kernels from the profiler).

    .venv/bin/python python/training/profile_ppo.py python/runs/watcher-runs-shallow/latest.pt [--games 512] [--steps 128]

First run 2026-10-27 on watcher-runs-shallow's Watcher A20 positions: choices between fights were 44% of the rows and
went through the usual pass, uncompiled, at 44,000 rows a second against the compiled fights'
120,800: more than half of PPO's time. Compiled since (`RunNet.choices_static`).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import time

import torch
import torch.nn.functional as F

from core.fight_model import masked
from core.fixed_shapes import CHOICE_CAPS, CHOICES, FIGHT, FIGHT_OPTIONS, OTHER, FixedShapes, kinds, minibatches
from core.sts_env import RunEnv
from training.train_runs import load

NAMES = {FIGHT: "fight", FIGHT_OPTIONS: "fight+options", OTHER: "other (usual pass)",
         **{c: f"choice {CHOICE_CAPS[c][0]}/{CHOICE_CAPS[c][1]}" for c in CHOICES}}
ALL = (FIGHT, FIGHT_OPTIONS, OTHER) + CHOICES

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--games", type=int, default=512)
p.add_argument("--steps", type=int, default=128)
p.add_argument("--minibatch", type=int, default=1024)
args = p.parse_args()

dev = torch.device("cuda")
net, ck = load(args.checkpoint, dev)
net.load_state_dict(ck["net"])
amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

# Real positions: the network plays Watcher A20 runs for --steps decisions.
t0 = time.time()
env = RunEnv(args.games, seed=4242, character="watcher", ascension_range=(20, 20))
net.eval()
I, Fl, M, A = [], [], [], []
for _ in range(args.steps):
    obs = env.observe()
    ints = torch.from_numpy(obs["ints"]).to(dev)
    fl = torch.from_numpy(obs["floats"]).to(dev)
    mask = torch.from_numpy(obs["mask"]).to(dev)
    with torch.no_grad(), amp():
        lg = masked(net(ints.long(), fl)[0].float(), mask)
    a = torch.distributions.Categorical(logits=lg).sample()
    env.step(a.cpu().numpy())
    I.append(ints.short()), Fl.append(fl), M.append(mask), A.append(a)
I, Fl, M, A = torch.cat(I), torch.cat(Fl), torch.cat(M), torch.cat(A)
B = I.shape[0]
P = torch.zeros(B, dtype=torch.bool, device=dev)
k = kinds(I)
print(f"{B:,} positions in {time.time() - t0:.0f} s; ints {I.shape[1]}, floats {Fl.shape[1]}")
for kind in ALL:
    if (k == kind).any():
        print(f"  {NAMES[kind]}: {int((k == kind).sum()):,} rows ({float((k == kind).float().mean()):.1%})")

from core.fight_model import INT_AT

sel = I[k == FIGHT_OPTIONS]
if len(sel):
    n_sel = (sel[:, INT_AT["option_card"][0]:INT_AT["option_card"][1]] > 0).sum(1).cpu()
    print("options in fights with a selection: percentiles 50/90/99 " + str([int(n_sel.float().quantile(q)) for q in (0.5, 0.9, 0.99)])
          + f", max {int(n_sel.max())}; at most 16: {float((n_sel <= 16).float().mean()):.1%}")

net.train()
fixed = FixedShapes(net, cuda_graphs=True)
opt = torch.optim.Adam(net.parameters(), lr=1e-6, eps=1e-5, fused=True)
with torch.no_grad(), amp():
    LP = torch.cat([F.log_softmax(masked(net(I[i:i + 4096].long(), Fl[i:i + 4096])[0].float(), M[i:i + 4096]), -1)
                    .gather(1, A[i:i + 4096, None]).squeeze(1) for i in range(0, B, 4096)])


def step(kind, idx, forward=None):
    opt.zero_grad(set_to_none=True)
    with amp():
        if forward is None:
            logits, value = fixed.forward(kind, I[idx].long(), Fl[idx], P[idx])
        else:
            logits, value = forward(I[idx].long(), Fl[idx], P[idx])
    logits, value = logits.float(), value.float()
    logp_all = F.log_softmax(masked(logits, M[idx]), dim=-1)
    logp = logp_all.gather(1, A[idx][:, None]).squeeze(1)
    ratio = (logp - LP[idx]).exp()
    adv = torch.randn_like(ratio)
    ent = -(logp_all.exp() * logp_all).sum(-1).mean()
    loss = torch.max(-adv * ratio, -adv * ratio.clamp(0.8, 1.2)).mean() + 0.5 * (value - torch.randn_like(value)).pow(2).mean() - 0.01 * ent
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
    opt.step()


plan = minibatches(k, args.minibatch)
by_kind = {kind: [idx for idx, kk in plan if kk == kind] for kind in ALL}
rate = {}
for kind, mbs in by_kind.items():
    if not mbs:
        continue
    for idx in mbs[:3]:  # compile and capture
        step(kind, idx)
    torch.cuda.synchronize()
    reps = [mbs[(3 + i) % len(mbs)] for i in range(20)]  # 20 timed steps, a small kind's minibatches again
    t = time.time()
    for idx in reps:
        step(kind, idx)
    torch.cuda.synchronize()
    dt = (time.time() - t) / len(reps)
    rate[kind] = len(reps[0]) / dt
    print(f"{NAMES[kind]:20} minibatch {len(reps[0]):5}: {dt * 1000:6.1f} ms a step, {rate[kind]:8,.0f} rows/s")

# An update of 262,144 positions, 2 epochs, in this mix.
share = {kind: float((k == kind).float().mean()) for kind in rate}
est = {kind: 2 * 262144 * share[kind] / rate[kind] for kind in rate}
print("estimated PPO seconds an update (262,144 positions, 2 epochs): "
      + ", ".join(f"{NAMES[kk]} {v:.2f}" for kk, v in est.items()) + f"; total {sum(est.values()):.2f}")

# Forward alone, compiled fight pass.
idx = by_kind[FIGHT][0]
with torch.no_grad(), amp():
    for _ in range(3):
        fixed.forward(FIGHT, I[idx].long(), Fl[idx], P[idx])
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        fixed.forward(FIGHT, I[idx].long(), Fl[idx], P[idx])
    torch.cuda.synchronize()
print(f"fight forward alone (no grad, compiled): {(time.time() - t) / 20 * 1000:.1f} ms")

# Op profile of the uncompiled fight pass (CUDA graphs hide their kernels from the profiler).
from torch.profiler import ProfilerActivity, profile

eager = lambda i, f, pr: net.combat_static(i, f, pr, False)
for _ in range(3):
    step(FIGHT, idx, eager)
torch.cuda.synchronize()
t = time.time()
for _ in range(5):
    step(FIGHT, idx, eager)
torch.cuda.synchronize()
print(f"fight step uncompiled: {(time.time() - t) / 5 * 1000:.1f} ms")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        step(FIGHT, idx, eager)
    torch.cuda.synchronize()
ka = prof.key_averages()
total_cuda = sum(e.self_device_time_total for e in ka) / 5 / 1000
print(f"GPU kernel time a fight step (uncompiled): {total_cuda:.1f} ms")
print(ka.table(sort_by="self_cuda_time_total", row_limit=30, max_name_column_width=60))
