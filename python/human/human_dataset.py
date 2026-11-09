"""Human decisions from won runs as the run network's inputs, on disk for supervised training.

    .venv/bin/python python/human/human_dataset.py ~/sts-data/ironclad/decisions --out ~/sts-data/human-won

Reads the decision shards `human_extract.py` wrote, keeps the decisions of won runs (or all with
--all), encodes them in parallel (`human_runs.encode`: the arrays `RunEnv.observe()` gives, for a
run rebuilt in that state on that screen) and writes flat arrays to <out>:
- ints.bin (int16, n x RUN_INT_LEN), floats.bin (float16, n x RUN_FLOAT_LEN), mask.bin (packed
  bits, n x ceil(ACTIONS / 8)): the network's inputs;
- slot.npy (the action slot the player took), kind.npy (0 card, 1 rest site, 2 grid card, 3 boss
  relic, 4 shop removal, 5 event), asc.npy (ascension), val.npy (held out: 1 run in 50, by play_id);
- how the run went from there, for the outcome head: floor.npy (the decision's floor), reached.npy
  (the floor the run ended on), won.npy and heart.npy (it won; it beat the Heart), from records
  `human_runs.decisions` wrote since 2026-10-27 (0 in older ones);
- meta.json: n, the row lengths and how many records the rebuilt state couldn't express.
--play-ids <file> keeps only the runs listed there (one play id a line; analysis/archetypes.py writes one archetype's winners), and
--min-floor and --max-floor only decisions on those floors: card picks learned from one archetype's players, rather than the average of them all.
With --lookup <fight network checkpoint> (one with a fight-outcome head), the card rewards' rows get the deck network's
lookup (core/lookup.py: each card's gain in that network's win chance and HP against this act's boss and the next act's
bosses), as training fills it (train_runs.py --lookup).
Then the rows are shuffled on disk into <out>/train and <out>/val (the same files, without val.npy),
with sequential reads and writes only (through buckets of rows), so training reads each batch as one
contiguous slice: random rows from a file this size would make it wait on the disk. The unshuffled
files are removed after. `pretrain_human.py` reads <out>/train and <out>/val with np.memmap.

    .venv/bin/python python/human/human_dataset.py --shuffle-only ~/sts-data/human-won   # redo just the shuffle
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("RAYON_NUM_THREADS", "1")  # one process per core already
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders
import human.human_runs as hr  # noqa: E402
from core.sts_env import ACTIONS, RUN_FLOAT_LEN, RUN_INT_LEN  # noqa: E402

KINDS = {"card": 0, "campfire": 1, "grid": 2, "boss": 3, "purge": 4, "event": 5}
F16_MAX = 65000.0


def held_out(play_id):
    return int(hashlib.md5(str(play_id).encode()).hexdigest()[:8], 16) % 50 == 0


PLAY_IDS = None  # --play-ids, in each worker


def read_play_ids(path):
    global PLAY_IDS
    if path:
        with open(path) as f:
            PLAY_IDS = {line.strip() for line in f if line.strip()}


def shard(args):
    path, everyone, min_asc, kinds, heart, min_floor, max_floor = args
    with gzip.open(path, "rt") as f:
        recs = [r for r in map(json.loads, f) if (r.get("heart") if heart else everyone or r.get("won"))
                and int(r.get("ascension") or 0) >= min_asc and (not kinds or r.get("kind") in kinds)
                and min_floor <= int(r.get("floor") or 0) <= max_floor and (PLAY_IDS is None or r.get("play_id") in PLAY_IDS)]
    if not recs:
        return None
    obs, slots, source, skipped = hr.encode(recs)
    if not len(slots):
        return None, len(skipped)
    # A rest site's smith or toke gives two samples: the choice, then the card on its grid. (An
    # event's second sample, such as the Golden Idol's escape, stays an event.)
    kinds = np.array([KINDS[recs[s]["kind"]] for s in source], dtype=np.int8)
    second = np.r_[False, source[1:] == source[:-1]] & (kinds == KINDS["campfire"])
    kinds[second] = KINDS["grid"]
    card_rows = np.flatnonzero(kinds == KINDS["card"])
    return {
        # (for --lookup: each card reward's row and its record)
        "card_recs": [(int(r), json.dumps(recs[source[r]])) for r in card_rows],
        "ints": obs["ints"].astype(np.int16),
        "floats": np.clip(obs["floats"], -F16_MAX, F16_MAX).astype(np.float16),
        "mask": np.packbits(obs["mask"], axis=1),
        "slot": slots.astype(np.int16),
        "kind": kinds,
        "asc": np.array([int(recs[s].get("ascension") or 0) for s in source], dtype=np.int8),
        "val": np.array([held_out(recs[s].get("play_id")) for s in source], dtype=bool),
        "floor": np.array([int(recs[s].get("floor") or 0) for s in source], dtype=np.int16),
        "reached": np.array([int(recs[s].get("floor_reached") or 0) for s in source], dtype=np.int16),
        "won": np.array([bool(recs[s].get("won")) for s in source], dtype=np.int8),
        "heart": np.array([bool(recs[s].get("heart")) for s in source], dtype=np.int8),
    }, len(skipped)


ROW_FILES = (("ints", np.int16, "ints"), ("floats", np.float16, "floats"), ("mask", np.uint8, "mask_bytes"))
SMALL = ("slot", "kind", "asc", "floor", "reached", "won", "heart")


def shuffle(out, rows_per_bucket=300_000, chunk=200_000, seed=0):
    """Shuffles <out>'s rows into <out>/train and <out>/val with sequential IO: a pass over the
    rows appends each to the bucket of its new position, then each bucket is sorted in memory and
    written out, buckets in order."""
    with open(os.path.join(out, "meta.json")) as f:
        meta = json.load(f)
    n = meta["n"]
    src = {k: np.memmap(os.path.join(out, f"{k}.bin"), dtype=dt, mode="r", shape=(n, meta[w])) for k, dt, w in ROW_FILES}
    small = {k: np.load(os.path.join(out, f"{k}.npy")) for k in SMALL if os.path.exists(os.path.join(out, f"{k}.npy"))}
    val = np.load(os.path.join(out, "val.npy"))
    rng = np.random.default_rng(seed)
    split = val.astype(np.int64)  # 0 train, 1 val
    pos = np.empty(n, dtype=np.int64)
    perms = []
    for s_ in (0, 1):
        perm = rng.permutation(np.flatnonzero(split == s_))
        pos[perm] = np.arange(len(perm))
        perms.append(perm)
    bucket = pos // rows_per_bucket
    tmp = os.path.join(out, "buckets")
    os.makedirs(tmp, exist_ok=True)
    name = lambda s_, b, k: os.path.join(tmp, f"{s_}-{b}-{k}.bin")  # noqa: E731
    keys = sorted(set(zip(split.tolist(), bucket.tolist())))
    handles = {(s_, b, k): open(name(s_, b, k), "wb") for s_, b in keys for k in ("ints", "floats", "mask", "pos")}
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        rows = {k: np.asarray(src[k][start:end]) for k, _, _ in ROW_FILES}
        sp, bk, ps = split[start:end], bucket[start:end], pos[start:end]
        for s_, b in set(zip(sp.tolist(), bk.tolist())):
            sel = (sp == s_) & (bk == b)
            for k in rows:
                rows[k][sel].tofile(handles[(s_, b, k)])
            ps[sel].tofile(handles[(s_, b, "pos")])
        print(f"  shuffle: bucketed {end}/{n} rows", flush=True)
    for h in handles.values():
        h.close()
    for s_, part in ((0, "train"), (1, "val")):
        d = os.path.join(out, part)
        os.makedirs(d, exist_ok=True)
        outs = {k: open(os.path.join(d, f"{k}.bin"), "wb") for k, _, _ in ROW_FILES}
        for b in sorted(b for x, b in keys if x == s_):
            p_ = np.fromfile(name(s_, b, "pos"), dtype=np.int64)
            order = np.argsort(p_)
            for k, dt, w in ROW_FILES:
                np.fromfile(name(s_, b, k), dtype=dt).reshape(-1, meta[w])[order].tofile(outs[k])
                os.remove(name(s_, b, k))
            os.remove(name(s_, b, "pos"))
        for f in outs.values():
            f.close()
        for k in small:
            np.save(os.path.join(d, f"{k}.npy"), small[k][perms[s_]])
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({**meta, "n": len(perms[s_])}, f, indent=1)
        print(f"  shuffle: {part} has {len(perms[s_])} rows", flush=True)
    os.rmdir(tmp)
    del src
    for k, _, _ in ROW_FILES:
        os.remove(os.path.join(out, f"{k}.bin"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("decisions", nargs="?", help="folder of decision shards (human_extract.py's <out>/decisions)")
    p.add_argument("--out")
    p.add_argument("--shuffle-only", metavar="OUT", help="only shuffle an already built <out> into train/ and val/")
    p.add_argument("--all", action="store_true", help="every kept run, not only won ones")
    p.add_argument("--heart", action="store_true", help="only runs that beat the Heart (records from 2026-10-27 on)")
    p.add_argument("--min-asc", type=int, default=0, help="only runs at this ascension or above (e.g. 20 for human-a20)")
    p.add_argument("--kinds", default=None, help="only these decision kinds (comma-separated, e.g. event)")
    p.add_argument("--play-ids", default=None, help="a file of play ids, one a line: only these runs")
    p.add_argument("--min-floor", type=int, default=0, help="only decisions from this floor on")
    p.add_argument("--max-floor", type=int, default=999, help="only decisions up to this floor")
    p.add_argument("--procs", type=int, default=max(os.cpu_count() - 2, 1))
    p.add_argument("--lookup", default=None, help="fight network checkpoint (with a fight-outcome head): fill the card rewards' lookup")
    p.add_argument("--lookup-draws", type=int, default=2)
    a = p.parse_args()
    if a.shuffle_only:
        return shuffle(a.shuffle_only)
    if not (a.decisions and a.out):
        p.error("pass the decisions folder and --out")
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(os.path.join(a.decisions, f) for f in os.listdir(a.decisions) if f.endswith(".jsonl.gz"))
    files = {k: open(os.path.join(a.out, f"{k}.bin"), "wb") for k in ("ints", "floats", "mask")}
    small = {k: [] for k in ("slot", "kind", "asc", "val", "floor", "reached", "won", "heart")}
    n = skipped = filled = 0
    with Pool(a.procs, initializer=read_play_ids, initargs=(a.play_ids,)) as pool:
        lk = None
        if a.lookup:
            # (after the workers fork: the network lives in this process, on the GPU if there is one)
            import contextlib
            import torch
            from core.lookup import Lookup, write
            from training.train_runs import load_fight_net
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else contextlib.nullcontext
            lk = Lookup(load_fight_net(a.lookup, device), device, amp, draws=a.lookup_draws)
        for i, res in enumerate(pool.imap(shard, [(p_, a.all, a.min_asc, a.kinds and a.kinds.split(","), a.heart, a.min_floor, a.max_floor) for p_ in paths], chunksize=4), 1):
            if res is not None:
                part, sk = res
                skipped += sk
                if part is not None:
                    if lk is not None and part["card_recs"]:
                        rows, recs_ = zip(*part["card_recs"])
                        for k, s in lk.human_cards(list(recs_)).items():
                            write(part["floats"][rows[k]], s)
                            filled += 1
                    for k in files:
                        part[k].tofile(files[k])
                    for k in small:
                        small[k].append(part[k])
                    n += len(part["slot"])
            if i % 500 == 0 or i == len(paths):
                print(f"{i}/{len(paths)} shards, {n} samples, {skipped} records left out" + (f", {filled} card rewards looked up" if lk else ""),
                      flush=True)
    for f in files.values():
        f.close()
    for k, v in small.items():
        np.save(os.path.join(a.out, f"{k}.npy"), np.concatenate(v))
    meta = {"n": n, "ints": RUN_INT_LEN, "floats": RUN_FLOAT_LEN, "actions": ACTIONS, "mask_bytes": (ACTIONS + 7) // 8,
            "skipped": skipped, "won_only": not a.all, "heart_only": a.heart, "kinds": KINDS,
            "lookup": a.lookup, "lookup_draws": a.lookup_draws if a.lookup else None, "looked_up": filled}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    kind = np.concatenate(small["kind"])
    val = np.concatenate(small["val"])
    print(f"{n} samples ({int(val.sum())} held out); by kind: " +
          ", ".join(f"{k} {int((kind == v).sum())}" for k, v in KINDS.items()))
    shuffle(a.out)


if __name__ == "__main__":
    sys.exit(main())
