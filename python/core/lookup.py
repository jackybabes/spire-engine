"""The deck network's lookup (jack, 2026-10-30: "once this is done we use it as lookup for deck network ... Whenever
decisions check the improvement to scores"): the fight network's fight-outcome head (run_model.RunNet.fight_outcome_pred)
read at the start of probe fights (crates/sts-py probe.rs: each option of a card reward, the deck as it would be, against
this act's boss and the next act's bosses, from a few redrawn openings), averaged over the openings.

    lk = Lookup(fight_net, device, amp)
    scores = lk.card_rewards(envs, games)        # training games at a card reward
    scores = lk.human_cards(records)             # human decision records (human_runs.py), kind "card"

Each is {source: Scores}: per option (-1 skipping, then each card offered) and per probe fight (0 this act's boss, then
the next act's), the head's mean win chance and HP share left. `Scores.gain()` is each option's improvement over
skipping.

`lk.fill(envs, obs)` writes it into an observation for the deck network (encoding 3: "opt_lookup", "run_lookup"; the
engine leaves them at zero), for the games at a card reward: each card option's gains over skipping in win chance and HP
share against this act's boss and against the next act's bosses (their mean), times GAIN_SCALE; and the deck as it
stands' own four scores. The deck network sees it when it chooses a card (train_runs.py --lookup).

`Lookup(..., every=True)` (train_runs.py --lookup-every) probes every choice that changes what a fight starts with, not
only card rewards (probe.rs `options`: a rest site's rest, lift and dig, Smith and Peace Pipe by their best card, a grid's
upgrade or removal, a shop's buys and its removal by its best card, a boss relic, a reward's relic or potion): the same
numbers for each option, in its option slot, against the run as it stands. `lk.options(envs, games)` scores them: one
engine call for every game (card rewards included), its rows and meta in reused buffers, one pass of the head. At the
map, a path into an elite or a hallway fight gets the head's win chance and HP share left against the act's elites or
strong hallway encounters (their means; the next two numbers zero): how dangerous that room is from where the run stands.
An event's options and Neow's: gains as elsewhere, each averaged over a few sampled outcomes (one per opening).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import time
from collections import defaultdict

import numpy as np
import torch

import sts
from core.fight_model import MAX_OPTIONS
from core.run_model import LOOKUP_FEATS, RUN_FLOAT_AT, RUN_INT_AT
from core.sts_env import LAYOUT, RUN_FLOAT_LEN, RUN_INT_LEN

# Gains are a few points of win chance: times this, about the size of the network's other inputs.
GAIN_SCALE = 10.0
CARD_REWARD = LAYOUT["screens"].index("card_reward") + 1  # encode_run.rs screen_id
# The screens Lookup(every=True) probes (probe.rs `probed`; a grid only if it takes one card it can upgrade, remove, bottle
# or copy: the engine decides).
PROBED = np.array([LAYOUT["screens"].index(k) + 1 for k in ("card_reward", "rest", "grid", "boss_relic", "shop", "rewards", "map",
                                                            "event", "neow")])
MAP = LAYOUT["screens"].index("map") + 1
NO_EVENTS = np.array([LAYOUT["screens"].index(k) + 1 for k in ("event", "neow")])
# The screens where train_runs.py logs how often the deck network takes the lookup's best option (Lookup.last): the
# option with the most win chance gained against this act's boss and the next act's; where taking nothing is a choice
# (skipping a card reward or a boss relic, leaving a shop), nothing if no option gains.
AGREE = {k: LAYOUT["screens"].index(k) + 1 for k in ("card_reward", "rest", "grid", "shop", "boss_relic", "event")}
SKIPPABLE = np.array([AGREE[k] for k in ("card_reward", "shop", "boss_relic")])
SCREEN_AT = RUN_INT_AT["screen"][0]
# The fight-outcome head reads a fight row's first KEEP_INTS ints and KEEP_FLOATS floats (the fight's fields, then the run
# token's: run_model.RunNet._run_token): the training games' probes move only those (Envs.probe_card_rewards_into keep).
KEEP_INTS = RUN_INT_AT["grid_purpose"][1]
KEEP_FLOATS = RUN_FLOAT_AT["run_scalars"][1]


def trained_events(state):
    """Whether a deck network's weights (a state dict) were trained with the lookup at events and Neow: its pick weights for
    those screens are non-zero (they start at zero and only move where the lookup is filled)."""
    w = state.get("lookup_pick.weight")
    return w is not None and bool(w[NO_EVENTS].abs().sum() > 0)


class Scores:
    """One decision's lookup: `win` and `hp` (options, fights), options in `options` (-1 first: skipping)."""

    def __init__(self, options, win, hp):
        self.options, self.win, self.hp = options, win, hp

    def gain(self, fights=None):
        """Each option's improvement over skipping in win chance and HP share, (options, fights) each (or the mean
        over the fights in `fights`, (options,))."""
        dw, dh = self.win - self.win[0], self.hp - self.hp[0]
        if fights is None:
            return dw, dh
        return dw[:, fights].mean(1), dh[:, fights].mean(1)


def write(row, s):
    """Writes Scores `s` into one encoded row (a float array of RUN_FLOAT_LEN, any float dtype): each card option's gains
    over skipping (times GAIN_SCALE) in "opt_lookup" and the deck as it stands' own scores in "run_lookup"."""
    (oa, ob), (ra, rb) = RUN_FLOAT_AT["opt_lookup"], RUN_FLOAT_AT["run_lookup"]
    dw, dh = s.gain()
    nxt = slice(1, None) if s.win.shape[1] > 1 else slice(0, 1)
    feats = np.stack([dw[:, 0], dh[:, 0], dw[:, nxt].mean(1), dh[:, nxt].mean(1)], 1) * GAIN_SCALE
    opt = np.zeros((MAX_OPTIONS, LOOKUP_FEATS), dtype=np.float32)
    opt[s.options[1:]] = feats[1:]  # each option in its slot (a card reward's card k is option k; skipping is the run token's)
    row[oa:ob] = opt.reshape(-1)
    row[ra:rb] = [s.win[0, 0], s.hp[0, 0], s.win[0, nxt].mean(), s.hp[0, nxt].mean()]


class Lookup:
    """`fixed` (core.fixed_shapes.FixedShapes of `net`): the head through its compiled passes (padded to a few sizes) and
    the training games' probes encoded into reused buffers (Envs.probe_card_rewards_into); without, the usual pass."""

    def __init__(self, net, device, amp, draws=4, seed=1, batch=4096, fixed=None, every=False, events=True):
        if net.fight_outcome_head is None:
            raise ValueError("the fight network has no fight-outcome head (train_runs.py --fight-outcome)")
        self.net, self.device, self.amp, self.draws, self.seed, self.batch = net, device, amp, draws, seed, batch
        self.fixed = fixed if fixed is not None and fixed.outcome_passes else None
        self.every = every
        # (events: also at events and Neow; off for a deck network trained before those were probed: trained_events)
        self.probed = PROBED if events else PROBED[~np.isin(PROBED, NO_EVENTS)]
        self._bufs, self._cap = None, 0
        # The last fill's games, the lookup's best option slots for each (S, MAX_OPTIONS), whether nothing is best, their screens.
        self.last = None
        # Where its time goes (train_runs.py logs them each update: take_times): the engine's probes, the head (rows to the
        # device included), the averaging and writing; the probe rows.
        self.times = dict.fromkeys(("probe_s", "head_s", "write_s", "rows"), 0.0)

    def take_times(self):
        """The time and rows since the last call."""
        t, self.times = self.times, dict.fromkeys(self.times, 0.0)
        return t

    def _head(self, ints, floats):
        """The head on probe rows (int64 and float32 tensors on the device): (n, 2) win chance and HP share, numpy."""
        with torch.no_grad(), self.amp():
            if self.fixed is not None:
                lg = self.fixed.fight_outcome(ints, floats)
            else:
                lg = torch.cat([self.net.fight_outcome_pred(ints[a:a + self.batch], floats[a:a + self.batch]).float()
                                for a in range(0, len(ints), self.batch)])
        return torch.sigmoid(lg.float()).cpu().numpy()

    def _rows(self, ib, fb, n, keep=None):
        """The first `n` rows of encoded arrays (bytes or a reused bytearray) as tensors on the device; with `keep` (ints,
        floats), rows of only that many (the rest of each row zero on the device)."""
        ki, kf = keep or (RUN_INT_LEN, RUN_FLOAT_LEN)
        i = torch.frombuffer(ib, dtype=torch.int32, count=n * ki).view(n, ki).to(self.device)
        f = torch.frombuffer(fb, dtype=torch.float32, count=n * kf).view(n, kf).to(self.device)
        if keep is None:
            return i.long(), f
        ints = torch.zeros(n, RUN_INT_LEN, dtype=torch.long, device=self.device)
        floats = torch.zeros(n, RUN_FLOAT_LEN, device=self.device)
        ints[:, :ki], floats[:, :kf] = i, f
        return ints, floats

    def _grid(self, meta, pred):
        """The probes' head outputs (n, 2) averaged per source, option and fight (meta: (source, option, fight, draw), or
        (source, option, sub, fight, draw) with an option's subs, the cards it could lead to: the option's scores are its
        best sub's, by its mean win chance over the fights): (sources, count, win, hp), the last three (sources, options
        + 1, fights), option -1 (the run as it stands) first, fights as probe.rs `fights` lists them. Only the cells
        probed are averaged (a few thousand a step), then laid out densely without the subs."""
        m = np.asarray(meta, dtype=np.int64).reshape(len(pred), -1)
        if m.shape[1] == 4:
            m = np.insert(m, 2, -1, axis=1)
        opt, sub, fight = m[:, 1] + 1, m[:, 2] + 1, m[:, 3]  # (option -1 first; sub -1, none, first)
        n_opt, n_sub, n_fight = int(opt.max()) + 1, int(sub.max()) + 1, int(fight.max()) + 1
        sources, s_idx = np.unique(m[:, 0], return_inverse=True)
        # Each probed cell (source, option, sub, fight): its probes' count and mean outputs.
        cells, inv = np.unique(((s_idx * n_opt + opt) * n_sub + sub) * n_fight + fight, return_inverse=True)
        cnt = np.bincount(inv)
        w, h = np.bincount(inv, weights=pred[:, 0]) / cnt, np.bincount(inv, weights=pred[:, 1]) / cnt
        f, oss = cells % n_fight, cells // n_fight  # oss: (source, option, sub)
        if n_sub > 1:
            # Each (source, option)'s best sub by its mean win chance over its fights (an option without subs has one).
            subs, inv2 = np.unique(oss, return_inverse=True)
            mean = np.bincount(inv2, weights=w) / np.bincount(inv2)
            so = subs // n_sub
            order = np.lexsort((-mean, so))
            best = subs[order[np.r_[True, so[order][1:] != so[order][:-1]]]]
            keep = np.isin(oss, best)
            f, oss, cnt, w, h = f[keep], oss[keep], cnt[keep], w[keep], h[keep]
        at = (oss // n_sub) * n_fight + f  # ((source, option), fight)
        shape = (len(sources), n_opt, n_fight)
        count, win, hp = np.zeros(shape, dtype=np.int64), np.zeros(shape), np.zeros(shape)
        count.reshape(-1)[at], win.reshape(-1)[at], hp.reshape(-1)[at] = cnt, w, h
        return sources, count, win, hp

    def _scores(self, meta, pred, aliases=()):
        """`_grid` as {source: Scores} (sources without a probe of the run as it stands left out); `aliases` (source,
        option, the option probed for it): a grid's copies of a card, given the copy probed's scores."""
        if not len(pred):
            return {}
        sources, count, win, hp = self._grid(meta, pred)
        out = {}
        for i, s in enumerate(sources.tolist()):
            opts = np.flatnonzero(count[i].any(1))
            fights = np.flatnonzero(count[i].any(0))
            if not len(opts) or opts[0] != 0:
                continue  # (no probe of the run as it stands: nothing to measure against)
            out[s] = Scores([int(o) - 1 for o in opts], win[i][np.ix_(opts, fights)], hp[i][np.ix_(opts, fights)])
        for g, k, k0 in aliases:
            sc = out.get(g)
            if sc is not None and k0 in sc.options:
                i = sc.options.index(k0)
                sc.options.append(k)
                sc.win, sc.hp = np.vstack([sc.win, sc.win[i:i + 1]]), np.vstack([sc.hp, sc.hp[i:i + 1]])
        return out

    @staticmethod
    def _feats(grid, aliases=()):
        """What `write` writes, for every source of `_grid` at once: (sources, opt_lookup rows (S, MAX_OPTIONS *
        LOOKUP_FEATS), run_lookup rows (S, LOOKUP_FEATS)), sources without a probe of the run as it stands left out."""
        sources, count, win, hp = grid
        has = count > 0
        ok = has[:, 0, 0]
        # Weights over the fights for "the next act's bosses": those probed; in Act 4 (the Heart alone) the Heart itself.
        nxt = has[:, 0, 1:]
        w = np.concatenate([~nxt.any(1)[:, None], nxt], 1).astype(np.float64)
        w /= w.sum(1, keepdims=True)
        nmean = lambda x: (x * w[:, None, :]).sum(2)
        dw, dh = win - win[:, :1], hp - hp[:, :1]
        feats = np.stack([dw[:, :, 0], dh[:, :, 0], nmean(dw), nmean(dh)], 2) * GAIN_SCALE
        feats[~has[:, :, 0]] = 0
        opt = np.zeros((len(sources), MAX_OPTIONS, LOOKUP_FEATS), dtype=np.float32)
        n = min(feats.shape[1] - 1, MAX_OPTIONS)
        opt[:, :n] = feats[:, 1:n + 1]  # option k in slot k
        probed = np.zeros((len(sources), MAX_OPTIONS), dtype=bool)
        probed[:, :n] = has[:, 1:n + 1, 0]
        if len(aliases):
            at = {int(g): i for i, g in enumerate(sources)}
            for g, k, k0 in aliases:
                if g in at:
                    opt[at[g], k], probed[at[g], k] = opt[at[g], k0], probed[at[g], k0]
        run = np.stack([win[:, 0, 0], hp[:, 0, 0], nmean(win)[:, 0], nmean(hp)[:, 0]], 1).astype(np.float32)
        return sources[ok], opt[ok].reshape(int(ok.sum()), MAX_OPTIONS * LOOKUP_FEATS), run[ok], probed[ok]

    def _buffers(self, n):
        """Room for `n` probe rows of KEEP_INTS and KEEP_FLOATS, and their meta (five int32 each)."""
        if n > self._cap:
            self._cap = max(n, 2 * self._cap)
            self._bufs = (bytearray(self._cap * KEEP_INTS * 4), bytearray(self._cap * KEEP_FLOATS * 4), bytearray(self._cap * 20))
        return self._bufs

    @staticmethod
    def _feats_path(grid, aliases=()):
        """The map's: each path option's mean win chance and HP share over its encounters (then two zeros), in its slot:
        (sources, opt_lookup rows (S, MAX_OPTIONS * LOOKUP_FEATS))."""
        sources, count, win, hp = grid
        has = count > 0
        n_f = np.maximum(has.sum(2), 1)
        feats = np.zeros(count.shape[:2] + (LOOKUP_FEATS,), dtype=np.float32)
        feats[:, :, 0] = (win * has).sum(2) / n_f
        feats[:, :, 1] = (hp * has).sum(2) / n_f
        opt = np.zeros((len(sources), MAX_OPTIONS, LOOKUP_FEATS), dtype=np.float32)
        n = min(feats.shape[1] - 1, MAX_OPTIONS)
        opt[:, :n] = feats[:, 1:n + 1]
        if len(aliases):
            at = {int(g): i for i, g in enumerate(sources)}
            for g, k, k0 in aliases:
                if g in at:
                    opt[at[g], k] = opt[at[g], k0]
        return sources, opt.reshape(len(sources), MAX_OPTIONS * LOOKUP_FEATS)

    def _probe_cards(self, envs, games):
        """The card rewards' probes of `games` through the head: (meta, pred), or None if none."""
        # At most about 7 options x 4 fights a card reward: room for 32 x draws rows a game (more if a call needs it).
        need = len(games) * 32 * self.draws
        keep = (KEEP_INTS, KEEP_FLOATS)
        t0 = time.perf_counter()
        try:
            ib, fb, _ = self._buffers(need)
            meta, n = envs.probe_card_rewards_into(games, self.draws, self.seed, ib, fb, keep)
        except ValueError:
            ib, fb, _ = self._buffers(8 * need)
            meta, n = envs.probe_card_rewards_into(games, self.draws, self.seed, ib, fb, keep)
        t1 = time.perf_counter()
        if not n:
            return None
        pred = self._head(*self._rows(ib, fb, n, keep))
        self.times["probe_s"] += t1 - t0
        self.times["head_s"] += time.perf_counter() - t1
        self.times["rows"] += n
        return meta, pred, ()

    def _probe_options(self, envs, games):
        """Every probed option's probes of `games` through the head: (meta, pred, aliases), or None if none."""
        keep = (KEEP_INTS, KEEP_FLOATS)
        t0 = time.perf_counter()
        bufs = self._buffers(max(self._cap, len(games) * 64 * self.draws))
        try:
            n, aliases = envs.probe_options_into(games, self.draws, self.seed, *bufs, keep)
        except ValueError as e:  # (more probes than room: the message names how many)
            bufs = self._buffers(int(str(e).rsplit(", ", 1)[1].split()[0]))
            n, aliases = envs.probe_options_into(games, self.draws, self.seed, *bufs, keep)
        t1 = time.perf_counter()
        if not n:
            return None
        ib, fb, mb = bufs
        pred = self._head(*self._rows(ib, fb, n, keep))
        self.times["probe_s"] += t1 - t0
        self.times["head_s"] += time.perf_counter() - t1
        self.times["rows"] += n
        return np.frombuffer(mb, dtype=np.int32, count=n * 5).reshape(n, 5), pred, aliases

    def card_rewards(self, envs, games):
        games = list(map(int, games))
        got = self._probe_cards(envs, games) if games else None
        return self._scores(*got) if got is not None else {}

    def options(self, envs, games):
        """Every probed option of the games in `games` (`every`'s lookup): {game: Scores}, options by their slots."""
        games = list(map(int, games))
        got = self._probe_options(envs, games) if games else None
        return self._scores(*got) if got is not None else {}

    def human_cards(self, records):
        meta, ib, fb = sts.probe_human_cards(records, self.draws, self.seed)
        if not meta:
            return {}
        return self._scores(meta, self._head(*self._rows(ib, fb, len(meta))))

    def fill(self, envs, obs):
        """Writes the lookup into `obs` (`RunEnv.observe`'s arrays of `envs`' games) for the games at a card reward (with
        `every`, at any screen it probes; see the module docs); returns `obs`, its floats copied first if they were
        read-only."""
        screen = obs["ints"][:, SCREEN_AT]
        games = np.flatnonzero(np.isin(screen, self.probed) if self.every else screen == CARD_REWARD)
        if not len(games):
            self.last = None
            return obs
        if not obs["floats"].flags.writeable:
            obs["floats"] = np.array(obs["floats"])
        floats = obs["floats"]
        self.last = None
        got = (self._probe_options if self.every else self._probe_cards)(envs, list(map(int, games)))
        if got is None:
            return obs
        t0 = time.perf_counter()
        meta, pred, aliases = got
        grid = self._grid(meta, pred)
        (oa, ob), (ra, rb) = RUN_FLOAT_AT["opt_lookup"], RUN_FLOAT_AT["run_lookup"]
        at_map = screen[grid[0]] == MAP
        if at_map.any():
            p_src, p_opt = self._feats_path(tuple(x[at_map] for x in grid), aliases)
            floats[p_src, oa:ob] = p_opt
            grid = tuple(x[~at_map] for x in grid)
        src, opt, run, probed = self._feats(grid, aliases)  # (all the games at once: what `write` writes)
        floats[src, oa:ob] = opt
        floats[src, ra:rb] = run
        # The best options by the win chance gained (this act's boss plus the next act's), for train_runs.py's log: every
        # option tied with the best (a grid's copies of a card), or nothing (-1) where taking nothing is the best choice.
        o = opt.reshape(len(src), MAX_OPTIONS, LOOKUP_FEATS)
        gain = np.where(probed, o[:, :, 0] + o[:, :, 2], -np.inf)
        top = gain.max(1, initial=-np.inf)
        scr = screen[src]
        nothing = ~probed.any(1) | (np.isin(scr, SKIPPABLE) & (top <= 0))
        best = probed & (gain >= top[:, None] - 1e-6) & ~nothing[:, None]
        self.last = (src, best, nothing, scr)
        self.times["write_s"] += time.perf_counter() - t0
        return obs
