"""How well the fight-outcome head (run_model.RunNet.fight_outcome_pred) calls fights before they start (jack, 2026-10-30:
the deck network is to ask it "with this deck, my chance against this boss").

    .venv/bin/python python/analysis/outcome_calibration.py <checkpoint> [--scenarios ~/sts-data/bosses-ironclad/eval.jsonl.gz]

Plays every held-out human fight greedily (train_runs.evaluate_scenarios' seeds) and reads the head at each fight's first
move (the deck, relics, potions and HP a human winner brought, the opening hand drawn): its win chance and HP share
against how the fight went. Reports, overall and by kind of fight: the mean chance against the win rate, the Brier
score (and a constant guess's, the kind's own win rate), how often a fight is called right, the ranking quality (AUC:
the chance a won fight was given a higher chance than a lost one), the HP share's error on won fights, and a
calibration table (fights in each tenth of predicted chance, and how many of them were won).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import contextlib
import json

import numpy as np
import torch

from core.fight_model import masked
from core.sts_env import RunEnv
from training.train_runs import SCENARIO_EVAL_SEED, load, read_scenarios


def kind(f):
    room = f.get("room", "boss")
    return "the Heart" if f["act"] == 4 and room == "boss" else f"Act {f['act']} {room}"


def auc(p, y):
    """The chance a positive was ranked above a negative (ties half)."""
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--scenarios", default="~/sts-data/bosses-ironclad/eval.jsonl.gz")
    ap.add_argument("--batch", type=int, default=1024)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else contextlib.nullcontext
    net, ck = load(a.checkpoint, device)
    net.load_state_dict(ck["net"])
    net.eval()
    if net.fight_outcome_head is None:
        sys.exit(f"{a.checkpoint} has no fight-outcome head (train_runs.py --fight-outcome)")
    lines = read_scenarios(os.path.expanduser(a.scenarios))
    fights = [json.loads(x) for x in lines]
    n, b = len(lines), a.batch
    env = RunEnv(b, seed=SCENARIO_EVAL_SEED, practice=1.0, max_steps=4000, character=ck["args"].get("character", "ironclad"))
    env.envs.set_scenarios(lines, 1.0, True)
    # Game g's episode e plays scenario e * b + g (sequential); its first observation is the fight's first move.
    episode, fresh = np.zeros(b, dtype=int), np.ones(b, dtype=bool)
    pred = np.full((n, 2), np.nan)
    won, hp = np.zeros(n), np.zeros(n)
    while (episode * b + np.arange(b) < n).any():
        obs = env.observe()
        ints = torch.from_numpy(obs["ints"]).to(device).long()
        floats = torch.from_numpy(obs["floats"]).to(device)
        prac = torch.ones(b, dtype=torch.bool, device=device)
        with torch.no_grad(), amp():
            logits, _ = net(ints, floats, prac)
            first = np.flatnonzero(fresh & (episode * b + np.arange(b) < n))
            if len(first):
                fo = torch.sigmoid(net.fight_outcome_pred(ints[first], floats[first]).float()).cpu().numpy()
                pred[episode[first] * b + first] = fo
        fresh[:] = False
        status = env.step(masked(logits.float(), torch.from_numpy(obs["mask"]).to(device)).argmax(-1).cpu().numpy())
        for g in np.flatnonzero(status):
            i = episode[g] * b + g
            if i < n:
                e = json.loads(env.last_episode(int(g)))
                won[i], hp[i] = float(e["won"]), e["hp"] / max(e["max_hp"], 1)
            episode[g] += 1
            fresh[g] = True
    ok = ~np.isnan(pred[:, 0])
    groups = {"all": ok} | {k: ok & np.array([kind(f) == k for f in fights]) for k in
                            ("Act 1 boss", "Act 2 boss", "Act 3 boss", "the Heart", "Act 1 elite", "Act 2 elite", "Act 3 elite")}
    print(f"{a.checkpoint} (update {ck.get('update')}): the head at each held-out human fight's first move ({ok.sum():,} fights)")
    print(f"{'':12} {'fights':>6} {'won':>6} {'chance':>7} {'Brier':>6} {'(const)':>7} {'right':>6} {'AUC':>5} {'HP err':>7}")
    for name, m in groups.items():
        p, y = pred[m, 0], won[m]
        if not len(p):
            continue
        hp_err = np.abs(pred[m, 1] - hp[m])[y == 1].mean() if (y == 1).any() else float("nan")
        print(f"{name:12} {len(p):6} {100 * y.mean():5.1f}% {100 * p.mean():6.1f}% {np.mean((p - y) ** 2):6.3f} {y.mean() * (1 - y.mean()):7.3f} "
              f"{100 * np.mean((p > 0.5) == y):5.1f}% {auc(p, y):5.2f} {hp_err:7.3f}")
    print("calibration (all fights): predicted chance -> fights, won")
    p, y = pred[ok, 0], won[ok]
    for lo in np.arange(0, 1, 0.1):
        m = (p >= lo) & (p < lo + 0.1 + (1e-9 if lo >= 0.9 else 0))
        if m.any():
            print(f"  {100 * lo:3.0f}-{100 * lo + 10:3.0f}%: {m.sum():5} fights, {100 * y[m].mean():5.1f}% won")


if __name__ == "__main__":
    main()
