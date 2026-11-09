"""Speed check and smoke test for the `sts` module: random play over many games.

    python/build.sh && python3 python/tests/bench.py [fight|run] [games] [steps]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import json
import random
import sys
import time

import sts

mode = sys.argv[1] if len(sys.argv) > 1 else "fight"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

envs = sts.Envs(n, mode=mode, seed=1)

# Random actions chosen in Rust (the engine's speed).
t = time.time()
done = 0
for k in range(steps):
    done += sum(1 for s in envs.step_random(k) if s)
dt = time.time() - t
print(f"{mode}: {n * steps / dt:,.0f} steps/s with step_random ({done} episodes finished)")

# Random actions chosen in Python (what a training loop pays per step, minus the network).
rng = random.Random(0)
t = time.time()
wins = losses = 0
for k in range(steps // 4):
    actions = [rng.randrange(c) for c in envs.num_legal()]
    for s in envs.step(actions):
        wins += s == 1
        losses += s == 2
dt = time.time() - t
print(f"{mode}: {n * (steps // 4) / dt:,.0f} steps/s with Python-chosen actions ({wins} won, {losses} lost)")
for i in range(n):
    last = envs.last_episode(i)
    if last:
        print("an episode that ended:", json.loads(last))
        break
