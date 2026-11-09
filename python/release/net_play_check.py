"""PyTorch's greedy play of fixed runs with a release network, one JSON line per run, to compare
with the Rust network's (`cargo run --release -p sts-net --example play -- <weights> <asc> <seed> <n>`).

    .venv/bin/python python/release/net_play_check.py web/models/ironclad-v1.safetensors 20 300000 32
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders
from release.export_release import load_release  # noqa: E402
from core.fight_model import masked  # noqa: E402
from core.sts_env import RunEnv  # noqa: E402


def main():
    weights, asc, first, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    net, meta = load_release(weights)
    env = RunEnv(n, seed=first, ascension=asc, max_steps=8000, character=meta.get("character", "ironclad"))
    done = [None] * n
    with torch.no_grad():
        while any(d is None for d in done):
            obs = env.observe()
            logits, _ = net(torch.from_numpy(obs["ints"].astype(np.int64)), torch.from_numpy(obs["floats"]))
            status = env.step(masked(logits, torch.from_numpy(obs["mask"])).argmax(-1).numpy())
            for i, s in enumerate(status):
                if s and done[i] is None:
                    e = json.loads(env.last_episode(i))
                    done[i] = {"seed": e["seed"], "won": e["won"], "act": e["act"], "floor": e["floor"], "hp": e["hp"], "steps": e["steps"]}
    for d in done:
        print(json.dumps(d))


if __name__ == "__main__":
    main()
