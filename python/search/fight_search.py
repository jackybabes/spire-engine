"""Look-ahead for fight moves by playouts (crates/sts-py/src/search.rs), for evaluation
(search_eval.py) and as a teacher in training (train_runs.py --search-weight).

From a fight move (not inside a card selection), each candidate first move (the network's `top`
likeliest legal moves, and ending the turn) is taken on `k` copies of the run whose unseen
randomness is redrawn, the same redraws for every candidate. Each copy is played on with the
network's own greedy moves until the player's turn comes round again (`turns` turns on), the
fight ends, or `max_moves` moves are made (the network can loop), and scored with the network's
value in float32 (in bfloat16 the value moves in steps about as large as the differences
between moves). A candidate's score is the average over its copies: no maximum over noisy
estimates, which favoured moves with many continuations in an earlier depth-first search.

A copy that died scores 0 (nothing more to earn), one that won the run a huge score, one whose
practice fight ended its practice reward (train_runs.practice_reward); all in the value's units
(rewards divided by the checkpoint's reward scale).

`value` says how a run's fights are scored. "run": by the run's value (where a copy whose fight
ended is scored on the reward screen, by the part of the network that makes choices between
fights). "fight": as practice fights are, by the practice value head mid-fight and by the
practice reward (0.5 + 0.5 x HP share + the potion bonus a potion kept) once the fight is won;
every leaf is then on one scale, the fight's. "outcome": as "fight", but a leaf mid-fight is scored by the fight-outcome
head (`RunNet.fight_outcome_pred`, trained alone on whether the network's play wins the fight and the HP share it ends
with: the practice reward's expectation, 0.5 x the win chance + 0.5 x the HP share, plus the potion bonus for each
potion held) instead of the practice value head.

`potion_value` (reward units, None = off) adds that much to a leaf for each potion still held,
outside boss fights (less what the practice reward already pays for one, with "fight"). Without
it the search drinks potions in hallway fights the network keeps for later: the fight's own score
pays 0.05 for one (a tenth of max HP), the run value hardly anything (ironclad-runs-habits-12 at A20: 2.5 potions a
run in hallway fights against the network's 0.5, fewer runs won).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import numpy as np
import torch

from core.fight_model import INT_AT, masked
from core.fixed_shapes import FIGHT, FIGHT_OPTIONS, FIGHT_ORBS, OPTIONS_CHUNK, kinds
from core.run_model import COMBAT, RUN_INT_AT
from core.sts_env import ACTIONS, LAYOUT, RUN_FLOAT_LEN, RUN_INT_LEN

END_TURN = LAYOUT["actions"]["end_turn"]
PICK = LAYOUT["actions"]["pick"][0]
CONFIRM = LAYOUT["actions"]["confirm"]
WON_RUN = 1e6
# With compiled passes (`fixed`), the copies' fight rows are padded to the smallest of these sizes that holds them (each
# size compiled once, with its CUDA graph), larger batches cut into the largest that fits the search's `batch`.
BUCKETS = (256, 1024, 4096, 16384)
SCREEN_AT, ROOM_AT = RUN_INT_AT["screen"][0], RUN_INT_AT["room"][0]
BOSS_ROOM = 7  # encode_run.rs room_id(RoomType::Boss)
FIGHT_POTIONS, RUN_POTIONS = INT_AT["potion"], RUN_INT_AT["run_potion"]


def potions_held(ints):
    """Potions held in each row of run arrays (the fight's slots in a fight, else the run's)."""
    fight = (ints[:, FIGHT_POTIONS[0]:FIGHT_POTIONS[1]] > 0).sum(1)
    run = (ints[:, RUN_POTIONS[0]:RUN_POTIONS[1]] > 0).sum(1)
    return np.where(ints[:, SCREEN_AT] == COMBAT, fight, run)


def reward_std(ck):
    """The checkpoint's reward scale (train_runs.py --scale-rewards): rewards were divided by it."""
    if not ck.get("args", {}).get("scale_rewards") or not ck.get("reward_scale"):
        return 1.0
    return float(np.sqrt(ck["reward_scale"][2] + 1e-8))


class FightSearch:
    def __init__(self, net, device, amp, k=8, top=4, turns=1, seed=1, scale=1.0, potion_bonus=0.0, max_moves=30, value="run",
                 potion_value=None, roots_per_call=256, batch=8192, fixed=None):
        assert value in ("run", "fight", "outcome")
        if value == "outcome" and net.fight_outcome_head is None:
            raise ValueError("value=\"outcome\" needs a fight-outcome head (train_runs.py --fight-outcome)")
        self.net, self.device, self.amp, self.value = net, device, amp, value
        # The network's compiled passes (core.fixed_shapes.FixedShapes), for the copies' moves; None: the usual pass.
        self.fixed = fixed
        # Reused buffers for the copies' arrays (Envs.playout_observe_into), and the rows they hold.
        self._bufs, self._cap = None, 0
        self.potion_value = potion_value
        self.k, self.top, self.turns, self.seed, self.max_moves = k, top, turns, seed, max_moves
        self.scale, self.potion_bonus = scale, potion_bonus
        self.roots_per_call, self.batch = roots_per_call, batch
        self.buckets = tuple(b for b in BUCKETS if b <= batch) or BUCKETS[:1]
        self.stats = {"searches": 0, "playouts": 0, "playout_moves": 0, "leaves": 0, "died": 0, "won_run": 0, "practice_over": 0}
        # Seconds by phase: setting the copies up, encoding them, the network's moves, stepping them, scoring the leaves
        # (encoding and the value pass); and the rounds of moves.
        self.times = dict.fromkeys(("start", "observe", "forward", "step", "results", "value"), 0.0)
        self.times["rounds"] = 0

    def candidates(self, mask, logits):
        """The first moves to try from a root with legal moves `mask` (None if there's no choice
        to make, or it's a card selection)."""
        if mask[PICK:CONFIRM + 1].any():
            return None
        legal = np.flatnonzero(mask)
        if len(legal) < 2:
            return None
        best = legal[np.argsort(-logits[legal], kind="stable")][:self.top]
        return sorted(set(best.tolist()) | ({END_TURN} if mask[END_TURN] else set()))

    def _forward(self, ints, floats, prac, values):
        """The network on rows of arrays, in batches: greedy moves (bfloat16), or values (float32)."""
        out = []
        own = lambda x: x if x.flags.writeable else np.array(x)  # (a reused buffer's rows need no copy)
        for a in range(0, len(ints), self.batch):
            i = torch.from_numpy(own(ints[a:a + self.batch])).to(self.device).long()
            f = torch.from_numpy(own(floats[a:a + self.batch])).to(self.device)
            p = torch.from_numpy(prac[a:a + self.batch]).to(self.device)
            with torch.no_grad():
                if values:
                    out.append(self.net(i, f, p)[1].float().cpu().numpy())
                else:
                    with self.amp():
                        out.append(self.net(i, f, p)[0].float())
        if values:
            return np.concatenate(out) if out else np.zeros(0, np.float32)
        return torch.cat(out) if out else None

    def scores(self, envs, ints, mask, logits, games, prac):
        """Look-ahead from each game in `games` (an `sts.Envs` of runs; `ints`, `mask` and `logits`
        its observation's integer arrays and legal moves and the network's logits, numpy; `prac`
        its practice mask). Returns {game: (first moves, their scores, standard errors)} for the
        games with a choice to make. A move's standard error is that of its score minus the
        network's own move's, copy by copy (every move plays on the same redrawn copies): how much
        of the gap could be the luck of the copies (0 for the network's own move)."""
        roots = []
        for g in games:
            c = self.candidates(mask[g], logits[g])
            if c is not None:
                roots.append((int(g), c))
        out = {}
        for at in range(0, len(roots), self.roots_per_call):
            chunk = roots[at:at + self.roots_per_call]
            # What each potion still held adds to a leaf of each root (value units).
            per_potion = np.zeros(len(chunk))
            if self.potion_value is not None:
                paid = self.potion_bonus if self.value in ("fight", "outcome") else 0.0
                rooms = np.array([ints[g, ROOM_AT] for g, _ in chunk])
                per_potion = np.where(rooms == BOSS_ROOM, 0.0, (self.potion_value - paid) / self.scale)
            own = [int(logits[g].argmax()) for g, _ in chunk]
            out.update(self._search(envs, chunk, prac, per_potion, own))
        return out

    def _outcome(self, ints, floats):
        """The fight-outcome head's expected practice reward for leaf rows (numpy), in value units: 0.5 x the win chance
        + 0.5 x the HP share left (float32)."""
        out = []
        own = lambda x: x if x.flags.writeable else np.array(x)
        for a in range(0, len(ints), self.batch):
            i = torch.from_numpy(own(ints[a:a + self.batch])).to(self.device).long()
            f = torch.from_numpy(own(floats[a:a + self.batch])).to(self.device)
            with torch.no_grad():
                p = torch.sigmoid(self.net.fight_outcome_pred(i, f).float())
            out.append((0.5 * p[:, 0] + 0.5 * p[:, 1]).cpu().numpy())
        return (np.concatenate(out) if out else np.zeros(0, np.float32)) / self.scale

    def _moves(self, ints, floats, prac):
        """The network's logits for the copies' rows (on the device), under the autocast: with `fixed`, each fight kind's
        rows through its compiled pass, padded to a size in self.buckets (a selection's in chunks of OPTIONS_CHUNK), any other
        row through the usual pass; without, the usual pass in batches."""
        if self.fixed is None:
            out = []
            with torch.no_grad(), self.amp():
                for a in range(0, len(ints), self.batch):
                    out.append(self.net(ints[a:a + self.batch], floats[a:a + self.batch], prac[a:a + self.batch])[0].float())
            return torch.cat(out)
        k = kinds(ints)
        out = torch.empty(len(ints), ACTIONS, device=ints.device)
        with torch.no_grad(), self.amp():
            for kind in torch.unique(k).tolist():
                rows = torch.nonzero(k == kind).squeeze(1)
                sizes = self.buckets if kind in (FIGHT, FIGHT_ORBS) else (OPTIONS_CHUNK,) if kind == FIGHT_OPTIONS else None
                if sizes is None:
                    out[rows] = self.net(ints[rows], floats[rows], prac[rows])[0].float()
                    continue
                for i in range(0, len(rows), sizes[-1]):
                    r = rows[i:i + sizes[-1]]
                    n = next(x for x in sizes if x >= len(r))
                    pad = r if len(r) == n else torch.cat([r, r[:1].expand(n - len(r))])
                    out[r] = self.fixed.passes[kind](ints[pad], floats[pad], prac[pad])[0][:len(r)].float()
        return out

    def _buffers(self, n):
        """Bytearrays with room for `n` copies' arrays (grown as needed)."""
        if n > self._cap:
            self._cap = max(n, 2 * self._cap)
            self._bufs = (bytearray(self._cap * RUN_INT_LEN * 4), bytearray(self._cap * RUN_FLOAT_LEN * 4), bytearray(self._cap * ACTIONS))
        return self._bufs

    def _search(self, envs, roots, prac, per_potion, own):
        fight = self.value in ("fight", "outcome")
        tm, t0 = self.times, time.perf_counter()
        info = np.asarray(envs.playout_start(roots, self.k, self.turns, self.seed, self.max_moves, fight), dtype=np.int64).reshape(-1, 3)
        tm["start"] += time.perf_counter() - t0
        root_of, first, det = info.T
        row_prac = np.asarray([fight or prac[g] for g, _ in roots], dtype=bool)[root_of]
        self.stats["searches"] += len(roots)
        self.stats["playouts"] += len(info)
        ib, fb, mb = self._buffers(len(info))
        while True:
            t0 = time.perf_counter()
            n, idx = envs.playout_observe_into(ib, fb, mb)
            if not n:
                tm["observe"] += time.perf_counter() - t0
                break
            tm["rounds"] += 1
            idx = np.asarray(idx)
            ints = torch.frombuffer(ib, dtype=torch.int32, count=n * RUN_INT_LEN).view(n, RUN_INT_LEN).to(self.device).long()
            floats = torch.frombuffer(fb, dtype=torch.float32, count=n * RUN_FLOAT_LEN).view(n, RUN_FLOAT_LEN).to(self.device)
            m = torch.frombuffer(mb, dtype=torch.uint8, count=n * ACTIONS).view(n, ACTIONS).to(self.device).bool()
            p = torch.from_numpy(row_prac[idx]).to(self.device)
            t1 = time.perf_counter()
            tm["observe"] += t1 - t0
            acts = masked(self._moves(ints, floats, p), m).argmax(-1).cpu().numpy()
            t2 = time.perf_counter()
            tm["forward"] += t2 - t1
            self.stats["playout_moves"] += len(acts)
            envs.playout_step(acts.tolist())
            tm["step"] += time.perf_counter() - t2
        t0 = time.perf_counter()
        meta, n_leaves = envs.playout_results_into(ib, fb)
        tm["results"] += time.perf_counter() - t0
        meta = np.asarray(meta, dtype=np.float64).reshape(-1, 6)
        kind = meta[:, 0].astype(int)
        for i, name in enumerate(("leaves", "died", "won_run", "practice_over")):
            self.stats[name] += int((kind == i).sum())
        v = np.zeros(len(meta))
        leaf = kind == 0
        ints = np.frombuffer(ib, dtype="<i4", count=n_leaves * RUN_INT_LEN).reshape(-1, RUN_INT_LEN)
        floats = np.frombuffer(fb, dtype="<f4", count=n_leaves * RUN_FLOAT_LEN).reshape(-1, RUN_FLOAT_LEN)
        t0 = time.perf_counter()
        if self.value == "outcome":
            v[leaf] = self._outcome(ints, floats) + self.potion_bonus * potions_held(ints) / self.scale
        else:
            v[leaf] = self._forward(ints, floats, row_prac[leaf], values=True)
        tm["value"] += time.perf_counter() - t0
        v[kind == 2] = WON_RUN
        practice = kind == 3
        held = np.zeros(len(meta))
        held[leaf] = potions_held(ints)
        if practice.any():
            won, hp, max_hp, potions, beaten = meta[practice, 1:].T
            reward = np.where(won > 0, 0.5 + 0.5 * hp / np.maximum(max_hp, 1) + self.potion_bonus * potions, 0.25 * np.minimum(beaten, 1))
            v[practice] = reward / self.scale
            held[practice] = np.where(won > 0, potions, 0)
        v += per_potion[root_of] * held
        out = {}
        for r, (g, cands) in enumerate(roots):
            rows = np.flatnonzero(root_of == r)
            if not len(rows):
                continue
            cands = np.asarray(cands)
            # Copy by copy: vals[move, copy].
            vals = np.zeros((len(cands), self.k))
            vals[np.searchsorted(cands, first[rows]), det[rows]] = v[rows]
            diff = vals - vals[int(np.flatnonzero(cands == own[r])[0])]
            se = diff.std(1, ddof=1) / np.sqrt(self.k) if self.k > 1 else np.zeros(len(cands))
            out[g] = (cands, vals.mean(1), se)
        return out
