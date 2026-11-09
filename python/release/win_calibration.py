"""The chance to win from a release network's run value: a logistic fit on its own greedy runs,
for the viewer's advice panel (web/networks.json `win`).

    .venv/bin/python python/release/win_calibration.py ironclad-v1.safetensors --runs 512 --ascensions 0,5,10,15,20

The run value is in training's reward units (the win reward plus habit rewards, scaled), not a
probability. At every decision of the fixed runs (seeds from 400,000, not the evaluation's) this
records the value, the floor and the ascension, and whether the run was won; then fits
P(won) = sigmoid(sum of TERMS, each times its weight) on 7 runs in 8 and checks it on the rest
(by predicted chance, ascension and act). Prints the weights as JSON (`web/app.js` `winChance`
sums the same terms). --save keeps the positions, --load refits from them without playing.
With --heart a run counts as won only if it beat the Heart (a network trained for the Heart, whose
Act 3 wins without the keys don't count); the JSON then has `"heart": true`, and the viewer says
"Chance to beat the Heart".
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders
from release.export_release import load_release  # noqa: E402
from core.fight_model import masked  # noqa: E402
from core.sts_env import RunEnv  # noqa: E402

SEED = 400_000


def play(net, device, runs, asc, batch, character="ironclad", heart=False):
    """Rows (value, floor, act, ascension, run id) of every decision, and each run's outcome
    (with `heart`, won only if it beat the Heart)."""
    rows, won = [], {}
    for first in range(0, runs, batch):
        n = min(batch, runs - first)
        env = RunEnv(n, seed=SEED + asc * 10_000 + first, ascension=asc, max_steps=8000, character=character)
        done = np.zeros(n, bool)
        with torch.no_grad():
            while not done.all():
                obs = env.observe()
                ints = torch.from_numpy(obs["ints"].astype(np.int64)).to(device)
                floats = torch.from_numpy(obs["floats"]).to(device)
                logits, value = net(ints, floats)
                sc = obs["f_run_scalars"]  # hp, max hp, gold, floor, act, ascension, ...
                live = np.flatnonzero(~done)
                v = value.float().cpu().numpy()
                for i in live:
                    rows.append((v[i], sc[i, 3], sc[i, 4], asc, (asc, first + i)))
                status = env.step(masked(logits, torch.from_numpy(obs["mask"]).to(device)).argmax(-1).cpu().numpy())
                for i in live:
                    if status[i]:
                        done[i] = True
                        ep = json.loads(env.last_episode(i))
                        won[(asc, first + i)] = ep["won"] and (not heart or ep["act"] >= 4)
    return rows, won


# The model's terms: name -> f(value, floor, ascension). web/app.js `winChance` has the same.
TERMS = {
    "bias": lambda v, f, a: np.ones_like(v),
    "value": lambda v, f, a: v,
    "floor": lambda v, f, a: f,
    "ascension": lambda v, f, a: a,
    "ascension2": lambda v, f, a: a * a,
    "value_ascension": lambda v, f, a: v * a,
    "value_floor": lambda v, f, a: v * f,
}


def features(v, floor, asc):
    return np.stack([t(v, floor, asc) for t in TERMS.values()], axis=1)


def fit(X, y, iters=50, l2=1e-6):
    """Logistic regression by Newton's method."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights")
    p.add_argument("--runs", type=int, default=512, help="per ascension")
    p.add_argument("--ascensions", default="0,5,10,15,20")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--save", help="keep the positions in this .npz")
    p.add_argument("--load", help="refit from a --save file instead of playing")
    p.add_argument("--heart", action="store_true", help="won = beat the Heart")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, meta = load_release(a.weights, device)
    if a.load:
        d = np.load(a.load)
        v, floor, act, asc, y, run = (d[k] for k in ("value", "floor", "act", "ascension", "won", "run"))
    else:
        rows, won = [], {}
        for asc_ in [int(x) for x in a.ascensions.split(",")]:
            r, w = play(net, device, a.runs, asc_, a.batch, meta.get("character", "ironclad"), a.heart)
            rows += r
            won.update(w)
            print(f"A{asc_}: {a.runs} runs, {len(r):,} decisions, won {np.mean(list(w.values())):.1%}", file=sys.stderr)
        v = np.array([r[0] for r in rows], np.float64)
        floor = np.array([r[1] for r in rows], np.float64)
        act = np.array([r[2] for r in rows], np.float64)
        asc = np.array([r[3] for r in rows], np.float64)
        y = np.array([won[r[4]] for r in rows], np.float64)
        run = np.array([r[4][0] * 100_000 + r[4][1] for r in rows])
        if a.save:
            np.savez_compressed(a.save, value=v, floor=floor, act=act, ascension=asc, won=y, run=run)
    held = run % 8 == 0
    X = features(v, floor, asc)
    w = fit(X[~held], y[~held])
    pr = 1 / (1 + np.exp(-X[held] @ w))
    yt = y[held]
    loss = -np.mean(yt * np.log(pr + 1e-12) + (1 - yt) * np.log(1 - pr + 1e-12))
    base = yt.mean()
    base_loss = -np.mean(yt * np.log(base) + (1 - yt) * np.log(1 - base))
    print(f"held-out decisions {held.sum():,}: log loss {loss:.4f} (constant {base_loss:.4f})", file=sys.stderr)
    print("by predicted chance (predicted / won / decisions):", file=sys.stderr)
    for lo in np.arange(0, 1, 0.1):
        m = (pr >= lo) & (pr < lo + 0.1)
        if m.sum():
            print(f"  {lo:.1f}-{lo + 0.1:.1f}: {pr[m].mean():.3f} / {yt[m].mean():.3f} / {m.sum():,}", file=sys.stderr)
    for name, key in (("ascension", asc[held]), ("act", act[held])):
        print(f"by {name}:", file=sys.stderr)
        for k in np.unique(key):
            m = key == k
            print(f"  {int(k)}: {pr[m].mean():.3f} / {yt[m].mean():.3f} / {m.sum():,}", file=sys.stderr)
    print(json.dumps({**{k: float(x) for k, x in zip(TERMS, w)}, **({"heart": True} if a.heart else {}),
                      "fit": f"{len(np.unique(run))} greedy runs of {meta.get('source')} update {meta.get('update')}, A{a.ascensions}; held-out log loss {loss:.3f} (constant {base_loss:.3f})"}))


if __name__ == "__main__":
    main()
