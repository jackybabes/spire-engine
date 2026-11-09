"""numpy views of the `sts` module's fight and run arrays.

    from core.sts_env import FightEnv
    env = FightEnv(256, seed=0)
    obs = env.observe()          # dict: field name -> array, plus "mask" (n, actions) bool
    status = env.step(actions)   # one action slot per game: 0 running, 1 won, 2 lost

    env = RunEnv(256, seed=0)     # whole runs: the fight fields (zero outside combat), then
    obs = env.observe()           # the run's ("i_run_relic", "i_opt_kind", "f_run_scalars", ...)

Fields and action slots are described by `sts.layout()` (see crates/sts-py/src/encode.rs).
Integer fields are IDs into embedding tables (0 = empty); float fields are raw numbers.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import numpy as np

import sts

LAYOUT = sts.layout()
INT_LEN = sum(n for _, n in LAYOUT["ints"])
FLOAT_LEN = sum(n for _, n in LAYOUT["floats"])
ACTIONS = LAYOUT["actions"]["count"]
RUN_INT_LEN = INT_LEN + sum(n for _, n in LAYOUT["run_ints"])
RUN_FLOAT_LEN = FLOAT_LEN + sum(n for _, n in LAYOUT["run_floats"])
ENCODING_VERSION = sts.ENCODING_VERSION  # raised whenever the network's inputs change (encode.rs)


def _split(row_major, fields):
    out, at = {}, 0
    for name, n in fields:
        out[name] = row_major[:, at:at + n]
        at += n
    return out


class FightEnv:
    MODE = "fight"
    INTS, FLOATS = LAYOUT["ints"], LAYOUT["floats"]
    INT_LEN, FLOAT_LEN = INT_LEN, FLOAT_LEN

    def __init__(self, n, seed=0, reuse_obs=False, **kwargs):
        """`reuse_obs`: `observe` writes into two sets of arrays in turn (`Envs.observe_into`),
        allocating nothing, so an observation's arrays hold until the next `observe` but one: for
        a rollout that keeps the last step's observation and no other."""
        self.envs = sts.Envs(n, mode=self.MODE, seed=seed, **kwargs)
        self.n = n
        sizes = (n * self.INT_LEN * 4, n * self.FLOAT_LEN * 4, n * ACTIONS)
        self._bufs = [tuple(bytearray(k) for k in sizes) for _ in range(2)] if reuse_obs else None
        self._turn = 0

    def observe(self):
        if self._bufs is None:
            ib, fb, mb = self.envs.observe_arrays()
            ints = np.frombuffer(ib, dtype="<i4").reshape(self.n, self.INT_LEN).copy()
            floats = np.frombuffer(fb, dtype="<f4").reshape(self.n, self.FLOAT_LEN).copy()
        else:
            ib, fb, mb = self._bufs[self._turn]
            self._turn ^= 1
            self.envs.observe_into(ib, fb, mb)
            ints = np.frombuffer(ib, dtype="<i4").reshape(self.n, self.INT_LEN)
            floats = np.frombuffer(fb, dtype="<f4").reshape(self.n, self.FLOAT_LEN)
        obs = {}
        obs.update({"i_" + k: v for k, v in _split(ints, self.INTS).items()})
        obs.update({"f_" + k: v for k, v in _split(floats, self.FLOATS).items()})
        obs["ints"] = ints
        obs["floats"] = floats
        obs["mask"] = np.frombuffer(mb, dtype=np.uint8).reshape(self.n, ACTIONS).astype(bool)
        return obs

    def step(self, actions):
        return np.asarray(self.envs.step_slots([int(a) for a in actions]), dtype=np.int8)

    def last_episode(self, i):
        return self.envs.last_episode(i)


class RunEnv(FightEnv):
    """Whole runs, with the same action slots as fights (see crates/sts-py/src/encode_run.rs)."""
    MODE = "run"
    INTS, FLOATS = LAYOUT["ints"] + LAYOUT["run_ints"], LAYOUT["floats"] + LAYOUT["run_floats"]
    INT_LEN, FLOAT_LEN = RUN_INT_LEN, RUN_FLOAT_LEN
