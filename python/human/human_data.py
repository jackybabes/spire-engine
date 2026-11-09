"""The human decisions dataset (`human_dataset.py`'s <out>/train or <out>/val) as memory maps,
read in contiguous batches; shared by pretrain_human.py and train_runs.py --human-data."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import json
import os
import queue
import threading

import numpy as np
import torch

from core.sts_env import ACTIONS, FLOAT_LEN, RUN_FLOAT_LEN, RUN_INT_LEN

# human_dataset.KINDS, by number.
KIND_NAMES = ["card", "rest", "grid", "boss", "purge", "event"]


OUTCOME = ("floor", "reached", "won", "heart")


class Human:
    """One split of the dataset (<data>/train or <data>/val, rows already shuffled on disk) as
    memory maps. Batches are contiguous slices, read in order from a random start each pass. A batch
    is (ints, floats, mask, slot, asc, kind, outcome): outcome (B, 4) is the decision's floor, the
    floor the run ended on, won, beat the Heart (`run_model.outcome_loss`), or None for a dataset
    from before 2026-10-27 (`self.outcome` None)."""

    def __init__(self, path, rng):
        with open(os.path.join(path, "meta.json")) as f:
            m = json.load(f)
        n = m["n"]
        self.ints = np.memmap(os.path.join(path, "ints.bin"), dtype=np.int16, mode="r", shape=(n, m["ints"]))
        self.floats = np.memmap(os.path.join(path, "floats.bin"), dtype=np.float16, mode="r", shape=(n, m["floats"]))
        self.mask = np.memmap(os.path.join(path, "mask.bin"), dtype=np.uint8, mode="r", shape=(n, m["mask_bytes"]))
        self.slot, self.kind, self.asc = (np.load(os.path.join(path, f"{k}.npy")) for k in ("slot", "kind", "asc"))
        files = [os.path.join(path, f"{k}.npy") for k in OUTCOME]
        self.outcome = np.stack([np.load(f).astype(np.float32) for f in files], 1) if all(map(os.path.exists, files)) else None
        if m["ints"] != RUN_INT_LEN:
            raise ValueError(f"{path}: rows have {m['ints']} ints, the encoding now {RUN_INT_LEN}; re-encode with human_dataset.py")
        # Rows encoded before the fight part grew (hand-info): every decision in the dataset is
        # outside a fight, where the fight part is all zeros, so the run part is kept and the
        # fight part padded with zeros.
        self.old_fight = m["floats"] - (RUN_FLOAT_LEN - FLOAT_LEN)
        if self.old_fight != FLOAT_LEN:
            sample = np.asarray(self.floats[: min(n, 4096), : self.old_fight])
            if self.old_fight < 0 or np.any(sample != 0):
                raise ValueError(f"{path}: rows have {m['floats']} floats, the encoding now {RUN_FLOAT_LEN}; re-encode with human_dataset.py")
        self.n = n
        self.path = path
        self.rng = rng
        self.at = int(rng.integers(n))
        self.passes = 0.0

    def kinds_bytes(self, kinds):
        """The memory `select_kinds(kinds)` would take."""
        row = 2 * self.ints.shape[1] + 2 * self.floats.shape[1] + self.mask.shape[1]
        return int(np.isin(self.kind, kinds).sum()) * row

    def select_kinds(self, kinds):
        """Only the rows of these kinds (KIND_NAMES indices), read into memory: batches then hold
        nothing else (train_runs.py --human-kinds event,boss read every kind and weighted the others
        to nothing, 83% of the rows passed through the network for no loss; 2026-10-28). For a
        subset small enough to hold (the Watcher's events and boss relics: 43,000 rows, 290 MB);
        use `next` without `start_prefetch` (which reads the files)."""
        idx = np.flatnonzero(np.isin(self.kind, kinds))
        self.ints, self.floats, self.mask = (np.asarray(x[idx]) for x in (self.ints, self.floats, self.mask))
        self.slot, self.kind, self.asc = self.slot[idx], self.kind[idx], self.asc[idx]
        if self.outcome is not None:
            self.outcome = self.outcome[idx]
        self.n = len(idx)
        self.at = int(self.rng.integers(max(self.n, 1)))
        return self.n

    def rows(self, a, b, device):
        mask = np.unpackbits(np.asarray(self.mask[a:b]), axis=1)[:, :ACTIONS].astype(bool)
        floats = np.asarray(self.floats[a:b]).astype(np.float32)
        if self.old_fight != FLOAT_LEN:
            padded = np.zeros((b - a, RUN_FLOAT_LEN), dtype=np.float32)
            padded[:, FLOAT_LEN:] = floats[:, self.old_fight:]
            floats = padded
        return (torch.from_numpy(np.asarray(self.ints[a:b]).astype(np.int64)).to(device),
                torch.from_numpy(floats).to(device),
                torch.from_numpy(mask).to(device),
                torch.from_numpy(self.slot[a:b].astype(np.int64)).to(device),
                torch.from_numpy(self.asc[a:b].astype(np.float32)).to(device),
                self.kind[a:b], self._outcome(a, b, device))

    def _outcome(self, a, b, device):
        return None if self.outcome is None else torch.from_numpy(self.outcome[a:b]).to(device)

    def start_prefetch(self, size, depth=8):
        """Read batches of `size` in a background thread (plain file reads, which let the GIL go),
        so the disk works while the GPU trains; `next(size)` then takes them from the queue. The
        thread owns `rng` and `at` from here on."""
        self._size = size
        self._q = queue.Queue(maxsize=depth)
        threading.Thread(target=self._fill, daemon=True).start()

    def _fill(self):
        files = {k: open(os.path.join(self.path, f"{k}.bin"), "rb", buffering=0) for k in ("ints", "floats", "mask")}
        widths = {"ints": (self.ints.shape[1], np.int16), "floats": (self.floats.shape[1], np.float16), "mask": (self.mask.shape[1], np.uint8)}

        def read(k, a, b):
            cols, dtype = widths[k]
            row = cols * np.dtype(dtype).itemsize
            buf = np.empty((b - a) * row, dtype=np.uint8)
            files[k].seek(a * row)
            files[k].readinto(buf)
            return buf.view(dtype).reshape(b - a, cols)

        size = self._size
        while True:
            if self.at + size > self.n:
                self.at = int(self.rng.integers(min(size, self.n)))
            a, b = self.at, self.at + size
            self.at = b
            mask = np.unpackbits(read("mask", a, b), axis=1)[:, :ACTIONS].astype(bool)
            floats = read("floats", a, b).astype(np.float32)
            if self.old_fight != FLOAT_LEN:
                padded = np.zeros((b - a, RUN_FLOAT_LEN), dtype=np.float32)
                padded[:, FLOAT_LEN:] = floats[:, self.old_fight:]
                floats = padded
            self._q.put((read("ints", a, b).astype(np.int64), floats, mask, self.slot[a:b].astype(np.int64),
                         self.asc[a:b].astype(np.float32), self.kind[a:b], (a, b), size / self.n))

    def next(self, size, device):
        if getattr(self, "_size", None) == size:
            ints, floats, mask, slot, asc, kind, (a, b), share = self._q.get()
            self.passes += share
            return (torch.from_numpy(ints).to(device), torch.from_numpy(floats).to(device), torch.from_numpy(mask).to(device),
                    torch.from_numpy(slot).to(device), torch.from_numpy(asc).to(device), kind, self._outcome(a, b, device))
        if self.at + size > self.n:
            self.at = int(self.rng.integers(min(size, self.n)))
        a = self.at
        self.at += size
        self.passes += size / self.n
        return self.rows(a, a + size, device)
