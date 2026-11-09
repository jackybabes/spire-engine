"""Checks for the human-outcome head (`RunNet(outcome=True)`, run_model.OutcomeHead), plain asserts,
no pytest needed:

    python python/tests/test_outcome.py

On positions between fights written as a human dataset (human_dataset.py's layout, with outcome
arrays that depend on the position), `human_data.Human` returns each batch's outcome and the head
learns it; the head changes nothing the network plays (the same logits and values with and without
it); a checkpoint with the head loads with it (`train_runs.load`), one without loads as before; and
a release leaves it out (`export_release.export`)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import tempfile

import numpy as np
import torch

from core.run_model import COMBAT, RUN_FLOAT_AT, RUN_INT_AT, RunNet, outcome_loss
from core.sts_env import ACTIONS, RUN_FLOAT_LEN, RUN_INT_LEN
from human.human_data import Human
from tests.test_static import positions


def write_split(d, ints, floats, mask, outcome):
    os.makedirs(d, exist_ok=True)
    n = len(ints)
    ints.astype(np.int16).tofile(os.path.join(d, "ints.bin"))
    floats.astype(np.float16).tofile(os.path.join(d, "floats.bin"))
    np.packbits(mask, axis=1).tofile(os.path.join(d, "mask.bin"))
    for k, v in (("slot", np.zeros(n, np.int16)), ("kind", np.zeros(n, np.int8)), ("asc", np.full(n, 20, np.int8))):
        np.save(os.path.join(d, f"{k}.npy"), v)
    for i, k in enumerate(("floor", "reached", "won", "heart")):
        np.save(os.path.join(d, f"{k}.npy"), outcome[:, i])
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"n": n, "ints": RUN_INT_LEN, "floats": RUN_FLOAT_LEN, "actions": ACTIONS, "mask_bytes": (ACTIONS + 7) // 8}, f)


def main():
    ints, floats, prac, legal = positions(steps=200, practice=0.0, character="watcher")
    between = ints[:, RUN_INT_AT["screen"][0]] != COMBAT
    ints, floats, legal = ints[between].numpy(), floats[between].numpy(), legal[between].numpy()
    # An outcome the position decides: the run's gold (a run scalar) splits the rows, and the
    # floors still to climb follow it.
    a, _ = RUN_FLOAT_AT["run_scalars"]
    gold = floats[:, a:a + 8]
    key = (gold * np.linspace(-1, 1, 8)).sum(1)
    good = key > np.median(key)
    floor = np.full(len(ints), 5.0)
    outcome = np.stack([floor, floor + np.where(good, 30, 3), good, good & (key > np.quantile(key, 0.75))], 1).astype(np.float32)

    with tempfile.TemporaryDirectory() as d:
        n = len(ints)
        cut = n * 4 // 5
        write_split(os.path.join(d, "train"), ints[:cut], floats[:cut], legal[:cut], outcome[:cut])
        write_split(os.path.join(d, "val"), ints[cut:], floats[cut:], legal[cut:], outcome[cut:])
        data = Human(os.path.join(d, "train"), np.random.default_rng(0))
        assert data.outcome is not None and data.outcome.shape == (cut, 4)
        b = data.next(256, "cpu")
        assert len(b) == 7 and b[6].shape == (256, 4)
        data.start_prefetch(256)
        b = data.next(256, "cpu")
        assert b[6].shape == (256, 4)

        torch.manual_seed(0)
        net = RunNet(d=32, layers=2, heads=2, outcome=True)
        opt = torch.optim.Adam(net.outcome.parameters(), lr=3e-3)  # the head alone: a fixed network's features
        held = Human(os.path.join(d, "val"), np.random.default_rng(1))
        vb = held.rows(0, held.n, "cpu")

        def val_loss():
            with torch.no_grad():
                return outcome_loss(net.outcome_pred(vb[0], vb[1]), vb[6])
        before, _ = val_loss()
        for _ in range(300):
            b = data.next(256, "cpu")
            loss, _ = outcome_loss(net.outcome_pred(b[0], b[1]), b[6])
            opt.zero_grad()
            loss.backward()
            opt.step()
        after, st = val_loss()
        assert float(after) < 0.7 * float(before), (float(before), float(after))
        print(f"outcome head learns: held-out loss {float(before):.3f} -> {float(after):.3f}, won called right {st['outcome_won_acc']:.0%} "
              f"({n} positions between fights)")

    # The head plays no part: the same weights without it give the same logits and values.
    plain = RunNet(d=32, layers=2, heads=2)
    missing, unexpected = plain.load_state_dict(net.state_dict(), strict=False)
    assert not missing and all(k.startswith("outcome.") for k in unexpected)
    i_all, f_all = torch.from_numpy(ints).long(), torch.from_numpy(floats)
    with torch.no_grad():
        l1, v1 = net(i_all, f_all)
        l2, v2 = plain(i_all, f_all)
    assert torch.equal(l1, l2) and torch.equal(v1, v2)
    print("the head changes no play: ok")

    from release.export_release import export, load_release
    from training.train_runs import load
    with tempfile.TemporaryDirectory() as d:
        args = {"d": 32, "layers": 2, "heads": 2, "outcome": True}
        torch.save({"net": net.state_dict(), "args": args}, os.path.join(d, "with.pt"))
        torch.save({"net": plain.state_dict(), "args": {**args, "outcome": False}}, os.path.join(d, "without.pt"))
        got, ck = load(os.path.join(d, "with.pt"), "cpu")
        got.load_state_dict(ck["net"])
        assert got.outcome is not None
        got, ck = load(os.path.join(d, "without.pt"), "cpu")
        got.load_state_dict(ck["net"])
        assert got.outcome is None
        meta = export(net, {"args": args}, os.path.join(d, "net.safetensors"))
        released, _ = load_release(os.path.join(d, "net.safetensors"))
        # (the file's parameters: a release leaves out the deck lookup's inputs, all zero)
    assert released.outcome is None and int(meta["parameters"]) == sum(p.numel() for n, p in released.named_parameters()
                                                                     if not n.startswith(("lookup_opt.", "lookup_run.", "lookup_pick.")))
    print("checkpoints and releases: ok")


if __name__ == "__main__":
    main()
