"""Offline search distillation (jack, 2026-10-30): a fight network trained towards the look-ahead's targets on
distill_data.py's searched positions, with a gentle, bounded loss over many passes; then the data can be searched
again with the better network (expert iteration).

    .venv/bin/python python/search/distill_train.py <checkpoint> <shard dir>... --out python/runs/<name> \\
        [--epochs 4] [--lr 2e-5] [--value 0.5] [--minibatch 1024] [--eval ~/sts-data/bosses-ironclad/eval.jsonl.gz]

Each searched position's target keeps the starting network's probabilities outside the search's candidates and
spreads the candidates' share by the search's target (distill_data.py: the network's odds times exp(shrunk gain /
temperature)), so where the search found no clear gain the target is the starting policy itself: the loss
KL(target || policy) is an anchor there and a pull towards the search's move only where the search is sure, and its
minimum is the target, so many passes can't flatten the policy (the on-policy attempts, train_runs.py
--search-weight, re-anchored the target on the moving policy every rollout and did). The value's squared error to
the returns (PPO's own targets, under the searched play) is weighed by --value. A twentieth of the positions are
held out: their loss, the share where the policy's likeliest move is the target's, and the policy's entropy are
logged each epoch. The held-out human fights (--eval, every one, greedy: train_runs.evaluate_scenarios) are played
at the start and after each epoch, and each epoch's network is saved in the checkpoint's format (update_<epoch>.pt
in --out; eval_worker.py and fight_losses.py take it).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import contextlib
import copy
import glob
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from core.fight_model import masked
from core.fixed_shapes import FixedShapes, kinds, minibatches
from training.train_runs import evaluate_scenarios, load, read_scenarios


def read_shards(dirs, device):
    """Every shard's arrays, joined, on `device` (ints stay int16)."""
    paths = sorted(p for d in dirs for p in glob.glob(os.path.join(os.path.expanduser(d), "shard_*.npz")))
    if not paths:
        sys.exit(f"no shards in {dirs}")
    parts = [dict(np.load(p)) for p in paths]
    data = {k: torch.from_numpy(np.concatenate([x[k] for x in parts])).to(device) for k in parts[0]}
    data["slot"] = data["slot"].long()
    return data, len(paths)


def start_logp(net, data, amp, chunk=4096):
    """The starting network's log-probabilities over the legal moves of every position (float16)."""
    out = torch.empty(data["mask"].shape, dtype=torch.float16, device=data["mask"].device)
    with torch.no_grad():
        for i in range(0, len(out), chunk):
            s = slice(i, i + chunk)
            with amp():
                lg, _ = net(data["ints"][s].long(), data["floats"][s], data["prac"][s])
            out[s] = F.log_softmax(masked(lg.float(), data["mask"][s]), -1).to(torch.float16)
    return out


def targets(logp0, slot, p, mask, sharpen=1.0):
    """The target over the legal moves: the starting probabilities, with the candidates' share spread by `p` (with
    `sharpen`, each candidate's log-odds gain over the starting policy, log p - log p0, times that: the search's
    temperature divided by it)."""
    t = logp0.float().exp() * mask
    valid = slot >= 0
    if sharpen != 1.0:
        lp0 = logp0.float().gather(1, slot.clamp(min=0))
        z = lp0 + sharpen * (p.clamp(min=1e-30).log() - lp0)
        p = torch.softmax(z.masked_fill(~valid, -1e9), -1) * valid
    # Unused candidate slots (-1) write to a spare column, dropped after (a real move's column could be 0).
    idx = torch.where(valid, slot, torch.full_like(slot, t.shape[1]))
    t = torch.cat([t, torch.zeros_like(t[:, :1])], 1)
    share = (t.gather(1, idx) * valid).sum(1, keepdim=True)
    t = t.scatter(1, idx, torch.where(valid, share * p, torch.zeros_like(p)))[:, :-1]
    return t / t.sum(1, keepdim=True).clamp(min=1e-12)


def losses(logits, value, rows, data, logp0, sharpen=1.0):
    """Mean KL(target || policy), the value's squared error, agreement with the target's likeliest move, entropy."""
    mask = data["mask"][rows]
    logp = F.log_softmax(masked(logits.float(), mask), -1)
    t = targets(logp0[rows], data["slot"][rows], data["p"][rows], mask, sharpen)
    kl = torch.where(t > 0, t * (t.clamp(min=1e-12).log() - logp), torch.zeros_like(t)).sum(-1).mean()
    v = (value.float() - data["ret"][rows]).pow(2).mean()
    agree = (logp.argmax(-1) == t.argmax(-1)).float().mean()
    ent = -(logp.exp() * logp).nan_to_num(0.0).sum(-1).mean()
    return kl, v, agree, ent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("data", nargs="+", help="distill_data.py output folders")
    p.add_argument("--out", required=True, help="run folder for the checkpoints and the log")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--value", type=float, default=0.5, help="weight of the value's squared error")
    p.add_argument("--sharpen", type=float, default=1.0,
                   help="the search's log-odds gains over the starting policy times this (its temperature divided by it)")
    p.add_argument("--minibatch", type=int, default=1024)
    p.add_argument("--max-grad", type=float, default=0.5)
    p.add_argument("--eval", default="~/sts-data/bosses-ironclad/eval.jsonl.gz", help="held-out human fights ('' to skip)")
    p.add_argument("--eval-batch", type=int, default=1024)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else contextlib.nullcontext
    net, ck = load(a.checkpoint, device)
    net.load_state_dict(ck["net"])
    character = ck["args"].get("character", "ironclad")
    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, "distill.jsonl")

    def write(row):
        print(json.dumps(row), flush=True)
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")

    t0 = time.time()
    data, shards = read_shards(a.data, device)
    n = len(data["act"])
    net.eval()
    logp0 = start_logp(net, data, amp)
    torch.manual_seed(0)
    perm = torch.randperm(n).to(device)
    held, train = perm[: n // 20], perm[n // 20:]
    changed = float((targets(logp0[held], data["slot"][held], data["p"][held], data["mask"][held], a.sharpen).argmax(-1)
                     != logp0[held].float().argmax(-1)).float().mean())
    write({"start": a.checkpoint, "shards": shards, "positions": n, "held_out": len(held), "target_moves_changed": changed,
           "read_s": round(time.time() - t0, 1), **{k: v for k, v in vars(a).items() if k != "data"}, "data": a.data})

    lines = read_scenarios(os.path.expanduser(a.eval)) if a.eval else []
    fixed = FixedShapes(net, cuda_graphs=device.type == "cuda")
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    row_kinds = kinds(data["ints"][train])

    def held_out():
        net.eval()
        with torch.no_grad():
            stats = []
            for i in range(0, len(held), 4096):
                r = held[i:i + 4096]
                with amp():
                    lg, v = net(data["ints"][r].long(), data["floats"][r], data["prac"][r])
                stats.append(torch.stack(losses(lg, v, r, data, logp0, a.sharpen)) * len(r))
        net.train()
        kl, v, agree, ent = (torch.stack(stats).sum(0) / len(held)).tolist()
        return {"held_kl": kl, "held_value_err": v, "held_agree": agree, "held_entropy": ent}

    def evaluate(epoch):
        if not lines:
            return {}
        t1 = time.time()
        row = evaluate_scenarios(net, device, lines, amp, batch=a.eval_batch, character=character)
        net.train()
        return {f"eval_{k}": v for k, v in row.items() if k not in ("set",)} | {"eval_s": round(time.time() - t1, 1)}

    write({"epoch": 0, **held_out(), **evaluate(0)})
    net.train()
    for epoch in range(1, a.epochs + 1):
        t1 = time.time()
        sums = torch.zeros(4, device=device)
        count = 0
        for rows, kind in minibatches(row_kinds, a.minibatch):
            r = train[rows]
            with amp():
                lg, v = fixed.forward(kind, data["ints"][r].long(), data["floats"][r], data["prac"][r])
            kl, verr, agree, ent = losses(lg, v, r, data, logp0, a.sharpen)
            loss = kl + a.value * verr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), a.max_grad)
            opt.step()
            sums += torch.stack([kl.detach(), verr.detach(), agree.detach(), ent.detach()])
            count += 1
        kl, verr, agree, ent = (sums / max(count, 1)).tolist()
        torch.save({**ck, "net": net.state_dict(), "update": epoch, "distill": {"from": a.checkpoint, "data": a.data, "epoch": epoch}},
                   os.path.join(a.out, f"update_{epoch}.pt"))
        write({"epoch": epoch, "train_kl": kl, "train_value_err": verr, "train_agree": agree, "train_entropy": ent,
               "train_s": round(time.time() - t1, 1), **held_out(), **evaluate(epoch)})


if __name__ == "__main__":
    main()
