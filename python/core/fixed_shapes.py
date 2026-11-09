"""The run network's fight rows through passes of fixed shape, compiled with CUDA graphs
(`train_runs.py --fixed-shapes`; branch cuda-graphs, 2026-10-24).

At d 128 the GPU spends most of a training step launching small kernels (it was about 10% busy).
`RunNet.combat_static` gives a fight row's pass a shape that doesn't depend on the batch (every orb
slot's token; all 64 option tokens or none), so torch.compile can capture the whole step as a CUDA
graph and replay it: on the Defect's defect-fights-0 positions the training pass ran 2.2 times as fast (34,500
positions a second eager, 74,900 fixed, compiled and graphed).

Each row is of one kind (`kinds`): a fight with no selection pending (the fixed pass without
option tokens; without orb tokens too if no orb is channelled, FIGHT, or with all 10 slots,
FIGHT_ORBS: 2026-10-27, a Watcher fight row went from 38 tokens to 28 and its training step from
8.0 ms to 5.65 a minibatch of 1,024, 7.0 of that from no longer building the option tokens), a
fight with one (the fixed pass with all 64), or a choice between fights, by the
caps its deck and options fit (`CHOICE_CAPS`: `RunNet.choices_static`, 2026-10-27; they went through
the usual pass, uncompiled, and with 44% of the Watcher's positions took more than half of PPO's
time: with these passes an update of watcher-runs-shallow's, 2,048 games x 128 steps at minibatch 1,024, went from
12.3 s to 8.7 s, PPO 7.6 to 4.3 s and the rollout's network 3.6 to 1.8 s, in 4.6 GB against 5.5;
`python/training/profile_ppo.py`). Training cuts each kind's rows into minibatches of one fixed size (`minibatches`; the rows
left over sit out that epoch, and a kind with fewer than a minibatch goes through the usual pass as
OTHER), so every minibatch is one kind. The rollout runs the whole batch through the pass without
options (with or without orbs, whichever more of its fights need) and the rows of the other kinds
again through theirs, in chunks (`FixedShapes.__call__`).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import torch
import torch.nn.functional as F

from core.fight_model import INT_AT, MAX_OPTIONS, masked
from core.run_model import COMBAT, MAX_DECK, RUN_INT_AT

FIGHT, FIGHT_OPTIONS, OTHER = 0, 1, 2
# Choices between fights, by the deck tokens (distinct cards) and option tokens they need: a kind
# for each pair of caps, the smallest a row fits. Watcher A20 runs by watcher-runs-shallow (69,300 choice rows,
# 2026-10-27): at most 21 distinct cards (99% at most 19); 2 options in the median row, 14 at the
# 99th percentile, 20 at most; 48 tokens a row with these caps, 44 with the usual pass's trimming.
CHOICE_CAPS = {3 + i * 3 + j: (deck, opt) for i, deck in enumerate((32, MAX_DECK)) for j, opt in enumerate((8, 16, MAX_OPTIONS))}
CHOICES = tuple(CHOICE_CAPS)
# A fight with no selection and an orb channelled (the Defect, or Prismatic Shard's slot): the
# pass with every orb slot's token; FIGHT rows have no orb and no orb tokens.
FIGHT_ORBS = 3 + len(CHOICES)
# The rollout's rows with a selection go through the pass with options in chunks of this many
# (about 3% of a 2,048-game batch: one chunk most steps), and its choices between fights in chunks
# of CHOICE_CHUNK (about 44% of the Watcher's: a few chunks a step).
OPTIONS_CHUNK = 128
CHOICE_CHUNK = 256
_OPTIONS = INT_AT["option_card"]
_ORBS = INT_AT["orb"]
_SCREEN = RUN_INT_AT["screen"][0]
_DECK = RUN_INT_AT["deck"]
_OPT_KIND = RUN_INT_AT["opt_kind"]


def kind_name(kind):
    """A kind's name in the logs: the choices' kinds are all "other"."""
    return {FIGHT: "fight", FIGHT_ORBS: "fight", FIGHT_OPTIONS: "options", None: "all"}.get(kind, "other")


def kinds(ints):
    """Each row's kind: FIGHT (no selection pending, no orb), FIGHT_ORBS (no selection, an orb),
    FIGHT_OPTIONS, or for a choice between fights the CHOICE_CAPS kind of the smallest caps its deck
    and options fit."""
    combat = ints[:, _SCREEN] == COMBAT
    options = (ints[:, _OPTIONS[0]:_OPTIONS[1]] > 0).any(1)
    orbs = (ints[:, _ORBS[0]:_ORBS[1]] > 0).any(1)
    n_deck = (ints[:, _DECK[0]:_DECK[1]] > 0).sum(1)
    n_opt = (ints[:, _OPT_KIND[0]:_OPT_KIND[1]] > 0).sum(1)
    choice = torch.full_like(n_deck, OTHER)
    for kind in reversed(CHOICES):  # the smallest caps last, so they win
        deck, opt = CHOICE_CAPS[kind]
        choice = torch.where((n_deck <= deck) & (n_opt <= opt), torch.full_like(choice, kind), choice)
    fight = torch.where(options, FIGHT_OPTIONS, torch.where(orbs, FIGHT_ORBS, FIGHT))
    return torch.where(combat, fight, choice).long()


def minibatch_size(kind, size):
    """A kind's minibatch: `size`, a quarter of it for fights with a selection (whose rows are
    about 2.5 times as long), and for choices at the largest deck and option caps (80 deck tokens and
    64 options, about 180 tokens a row with the unblended network's relics), half of it at one of
    them (2026-10-30: ironclad-deck-1, 4 x 256 unblended, ran out of its 10 GB on them)."""
    if kind == FIGHT_OPTIONS:
        return max(size // 4, 1)
    if kind in CHOICE_CAPS:
        deck, opt = CHOICE_CAPS[kind]
        big = (deck == MAX_DECK) + (opt == MAX_OPTIONS)
        return max(size // (1, 2, 4)[big], 1)
    return size


def minibatches(k, size, generator=None):
    """One epoch's minibatches over rows of kinds `k`, as (rows, kind): each fixed kind's rows
    shuffled and cut into minibatches of a fixed size (`minibatch_size`). The rows left over
    (fewer than a minibatch, a different few each epoch) sit out this epoch: putting them through
    the usual pass would cost its memory beside the CUDA graphs' (2026-10-24: 9 GB was too little
    with both at 2,048 games). A kind with fewer rows than one minibatch goes through the usual
    pass, as do any OTHER rows, shuffled together and cut into `size` (kind OTHER). The
    minibatches in random order."""
    out, rest = [], []
    for kind in (FIGHT, FIGHT_ORBS, FIGHT_OPTIONS) + CHOICES:
        n = minibatch_size(kind, size)
        rows = torch.nonzero(k == kind).squeeze(1)
        rows = rows[torch.randperm(len(rows), device=rows.device, generator=generator)]
        full = len(rows) // n * n
        out += [(rows[i:i + n], kind) for i in range(0, full, n)]
        if full == 0:
            rest.append(rows)
    rest.append(torch.nonzero(k == OTHER).squeeze(1))
    rest = torch.cat(rest)
    rest = rest[torch.randperm(len(rest), device=rest.device, generator=generator)]
    out += [(rest[i:i + size], OTHER) for i in range(0, len(rest), size)]
    order = torch.randperm(len(out), generator=generator).tolist()
    return [out[i] for i in order]


class FixedShapes:
    """The fixed passes (compiled; with CUDA graphs on a GPU) and the usual one, for `net`."""

    def __init__(self, net, compile=True, cuda_graphs=True):
        self.net = net
        self._orig_mod = net  # what fight_search.py and the like take apart

        def fight(ints, floats, practice):
            return net.combat_static(ints, floats, practice, False, orbs=False)

        def fight_orbs(ints, floats, practice):
            return net.combat_static(ints, floats, practice, False)

        def fight_options(ints, floats, practice):
            return net.combat_static(ints, floats, practice, True)

        def choices(caps):
            return lambda ints, floats, practice: net.choices_static(ints, floats, *caps)

        mode = "reduce-overhead" if cuda_graphs else None
        wrap = (lambda f: torch.compile(f, mode=mode, dynamic=False)) if compile else (lambda f: f)
        self.passes = {FIGHT: wrap(fight), FIGHT_ORBS: wrap(fight_orbs), FIGHT_OPTIONS: wrap(fight_options), OTHER: net}
        for kind, caps in CHOICE_CAPS.items():
            self.passes[kind] = wrap(choices(caps))

        # The fight-outcome head (the deck network's lookup) on fight rows with no selection pending, if the network has it.
        self.outcome_passes = {}
        if getattr(net, "fight_outcome_head", None) is not None:
            self.outcome_passes = {FIGHT: wrap(lambda ints, floats: net.fight_outcome_static(ints, floats, orbs=False)),
                                   FIGHT_ORBS: wrap(lambda ints, floats: net.fight_outcome_static(ints, floats))}

    def fight_outcome(self, ints, floats, buckets=(256, 1024, 4096)):
        """The fight-outcome head's (B, 2) logits for fight rows: each kind with a compiled pass (FIGHT, FIGHT_ORBS)
        padded to the smallest of `buckets` that holds its rows (more cut into the largest; each size compiled once),
        any other row through `fight_outcome_pred`. Call without gradients, under the same autocast each time."""
        k = kinds(ints)
        out = torch.empty(len(ints), 2, device=ints.device)
        for kind in torch.unique(k).tolist():
            rows = torch.nonzero(k == kind).squeeze(1)
            if kind not in self.outcome_passes:
                out[rows] = self.net.fight_outcome_pred(ints[rows], floats[rows]).float()
                continue
            for i in range(0, len(rows), buckets[-1]):
                r = rows[i:i + buckets[-1]]
                n = next(b for b in buckets if b >= len(r))
                pad = r if len(r) == n else torch.cat([r, r[:1].expand(n - len(r))])
                out[r] = self.outcome_passes[kind](ints[pad], floats[pad])[:len(r)].float()
        return out

    def eval(self):
        self.net.eval()
        return self

    def train(self, mode=True):
        self.net.train(mode)
        return self

    def forward(self, kind, ints, floats, practice):
        """A minibatch of one kind (from `minibatches`). The outputs are copies: a CUDA graph's
        are overwritten by its next replay."""
        logits, value = self.passes[kind](ints, floats, practice)
        return (logits.clone(), value.clone()) if kind != OTHER else (logits, value)

    def logp(self, ints, floats, practice, mask, act, size):
        """The taken moves' log-probabilities under the network now, for a whole batch without
        gradients (the network at an update's start, `train_runs.py --proximal`): each fixed kind's
        rows in chunks of its minibatch size (`size`, a quarter of it with a selection; the last
        chunk padded with its own first rows), the rest through the usual pass, one chunk at a
        time: `ints` may be the rollout's int16, made long a chunk at a time (the whole batch's at
        once, 2.2 GB, ran out of GPU memory in defect-fights-0-proximal, 2026-10-24)."""
        out = torch.empty(len(ints), dtype=torch.float32, device=ints.device)
        k = kinds(ints)
        for kind in (FIGHT, FIGHT_ORBS, FIGHT_OPTIONS, OTHER) + CHOICES:
            n = minibatch_size(kind, size)
            rows = torch.nonzero(k == kind).squeeze(1)
            for i in range(0, len(rows), n):
                r = rows[i:i + n]
                pad = r if kind == OTHER or len(r) == n else torch.cat([r, r[:1].expand(n - len(r))])
                lg, _ = self.passes[kind](ints[pad].long(), floats[pad], practice[pad])
                lg = masked(lg[:len(r)].float(), mask[r])
                out[r] = F.log_softmax(lg, -1).gather(1, act[r, None]).squeeze(1)
        return out

    def __call__(self, ints, floats, practice, deck=None):
        """The whole (fixed-size) rollout batch: every row through the pass without options, with
        orb tokens or without, whichever more of its fights need, then the fights needing the other
        in chunks of CHOICE_CHUNK, the rows with a selection through the pass with options in
        chunks of OPTIONS_CHUNK (the last padded with its own first rows; 2026-10-24: they went
        through the usual pass, uncompiled), the choices between fights through their kinds'
        passes in chunks of CHOICE_CHUNK (likewise; 2026-10-27), and any rest through the usual
        pass, their outputs put in. With `deck` (another FixedShapes: the split, core.split), the
        choices between fights and the rest (never a fight row) go through `deck`'s passes instead."""
        choices_by = self if deck is None else deck
        k = kinds(ints)
        # The rows by kind, with one wait for the GPU (each kind's own nonzero waited, nine a step:
        # 8.7 ms, more than the fight pass's 3.7; 2026-10-27).
        order = torch.argsort(k, stable=True)
        counts = torch.bincount(k, minlength=FIGHT_ORBS + 1).tolist()
        starts = [sum(counts[:j]) for j in range(len(counts))]
        whole, other = (FIGHT_ORBS, FIGHT) if counts[FIGHT_ORBS] > counts[FIGHT] else (FIGHT, FIGHT_ORBS)
        logits, value = self.passes[whole](ints, floats, practice)
        logits, value = logits.clone(), value.clone()
        for kind, chunk in ((other, CHOICE_CHUNK), (FIGHT_OPTIONS, OPTIONS_CHUNK)) + tuple((c, CHOICE_CHUNK) for c in CHOICES):
            rows = order[starts[kind]:starts[kind] + counts[kind]]
            passes = choices_by.passes if kind in CHOICES else self.passes
            for i in range(0, len(rows), chunk):
                r = rows[i:i + chunk]
                pad = r if len(r) == chunk else torch.cat([r, r[:1].expand(chunk - len(r))])
                l2, v2 = passes[kind](ints[pad], floats[pad], practice[pad])
                logits[r] = l2[:len(r)].to(logits.dtype)
                value[r] = v2[:len(r)].to(value.dtype)
        rest = order[starts[OTHER]:starts[OTHER] + counts[OTHER]]
        if len(rest):
            l2, v2 = choices_by.net(ints[rest], floats[rest], practice[rest])
            logits[rest] = l2.to(logits.dtype)
            value[rest] = v2.to(value.dtype)
        return logits, value
