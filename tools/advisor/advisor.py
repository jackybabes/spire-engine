"""In-game advisor: shows what the run network would do, in a box in the corner of the real game.

CommunicationMod (with this repo's patch) runs this as its external command and sends it every
game state; for each one this answers `advice <JSON list of lines>`, which the patch draws in the
top-left corner (F9 hides it). It never sends a game action, so you play as usual.

    tools/advisor/setup.sh [checkpoint]     # point CommunicationMod at this, with a checkpoint
    # then launch the harness game (see README) and play an Ironclad run

The network's inputs come from `sts.advise` (crates/sts-py/src/advise.rs): the game's JSON turned
into the same arrays as in training. Log: harness/advisor/advisor.log.
"""
import argparse
import json
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
OUT_DIR = os.path.join(REPO, "harness", "advisor")
TOP = 3


def send(lines):
    print("advice " + json.dumps(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=os.path.join(OUT_DIR, "model.pt"))
    args = p.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    log = open(os.path.join(OUT_DIR, "advisor.log"), "a", buffering=1)

    def note(*a):
        print(time.strftime("%H:%M:%S"), *a, file=log)

    # CommunicationMod waits a few seconds for this; load the network after.
    print("ready", flush=True)
    import numpy as np
    import torch

    import sts
    from core.fight_model import masked
    from core.sts_env import RUN_FLOAT_LEN, RUN_INT_LEN
    from training.train_runs import load

    torch.set_num_threads(2)
    net, ck = load(args.checkpoint, torch.device("cpu"))
    net.load_state_dict(ck["net"])
    net.eval()
    note("loaded", args.checkpoint, "update", ck.get("update"))
    send([f"Network advisor ready ({os.path.basename(args.checkpoint)}, update {ck.get('update')})"])

    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if "error" in msg:
            continue
        gs = msg.get("game_state")
        if not msg.get("in_game") or not gs:
            send(["Network advisor: waiting for a run"])
            continue
        try:
            if gs.get("class") not in (None, "IRONCLAD"):
                send(["Network advisor: it only plays the Ironclad"])
                continue
            ib, fb, mb, labels = sts.advise(json.dumps(gs), msg.get("available_commands") or [])
            if not labels:
                send(["Network: nothing to choose here"])
                continue
            ints = torch.from_numpy(np.frombuffer(ib, dtype="<i4").reshape(1, RUN_INT_LEN).astype(np.int64))
            floats = torch.from_numpy(np.frombuffer(fb, dtype="<f4").reshape(1, RUN_FLOAT_LEN).copy())
            mask = torch.from_numpy(np.frombuffer(mb, dtype=np.uint8).reshape(1, -1).astype(bool))
            with torch.no_grad():
                logits, _ = net(ints, floats)
            probs = torch.softmax(masked(logits.float(), mask), -1)[0].numpy()
            # Identical choices (two Strikes at the same monster) are one choice to a player.
            by_text = {}
            for k, text in labels:
                text = " ".join(text.replace("[", " ").replace("]", " ").split())  # Neow and event buttons
                by_text[text] = by_text.get(text, 0.0) + float(probs[k])
            best = sorted(by_text, key=lambda t: -by_text[t])[:TOP]
            fight = gs.get("room_phase") == "COMBAT" and gs.get("screen_type") in ("NONE", "HAND_SELECT", "GRID", "CARD_REWARD")
            where = "fight" if fight else gs.get("screen_type", "").lower().replace("_", " ")
            title = f"Network suggests  (floor {gs.get('floor')}, {where})"
            out = [title] + [f"{i + 1}. {t}   {by_text[t]:.0%}" for i, t in enumerate(best)]
            send(out)
            note(gs.get("floor"), gs.get("screen_type"), " | ".join(out[1:]))
        except Exception as e:  # noqa: BLE001 - show it in the box rather than dying
            note("error", repr(e), traceback.format_exc())
            send(["Network advisor error:", repr(e)[:80]])


if __name__ == "__main__":
    main()
