"""Overnight real-game check: play back a batch of engine runs and log every one that differs.

Run it through tools/trace/soak.sh (which sets CommunicationMod's command, keeps the Mac awake
and restarts the game if it crashes or freezes):

    tools/trace/soak.sh python/results/soak/batch.json.gz

or set CommunicationMod's command by hand:

    command=/opt/homebrew/bin/python3 -I <repo>/tools/trace/soak.py <batch.json.gz> <out dir> [delay seconds]

The batch comes from python/watch/soak_runs.py: runs in playback.py's format plus how each ended in
the engine. soak.sh doesn't trust the game commands saved in it: it works them out again from
each run's seed and decisions with the current engine (`sts-cli commands`, into <out dir>/batch.json),
so a fix to the engine's commands applies to batches made before it. Each run is started on its seed and its decisions are sent one by one (screens are
handled as in playback.py), with the game at harness speed 6. A run stops as soon as the game
and the engine differ:
- differs: the game's floor or HP before a decision isn't the engine's, still after letting the
  game run a few times 30 frames (it applies some effects, like an obtained card, after it reports);
- refused: the game keeps refusing a command;
- ended_early: the game's run ended (death or victory) before the engine's decisions ran out;
- result_differs: both ended, but one won and the other didn't;
- stuck: no progress for STUCK_SECONDS (a screen playback doesn't know how to get past, or the game
  waiting for something);
- crashed: the game died twice while playing this run;
- engine_changed: not played; the current engine can't take the saved decisions any more (soak.sh
  works out every run's commands, floors and HP with the current engine first, `sts-cli commands`).
Otherwise the run is `matched`. The run is then abandoned and the next one starts. A run that
didn't match is first played again from the start (SOAK_RETRIES, default 1): if a retry matches it
is recorded as `flaky`, since the modded game sometimes misses a monster's hit with nothing wrong
in the engine; one that fails every time keeps its status.

Output in <out dir>:
- summary.jsonl: one line per run (status, seed, step reached, game and engine state, detail);
- logs/<n>-<seed>-interrupted-<time>.log: a run cut short by a game restart (froze or crashed);
- logs/<n>-<seed>-<status>[-try<k>].log: the full playback log of every try that didn't match, and
  of a retry that did (the same lines playback.py writes, with the game's hand or deck, and in
  combat its monsters, at each step); tries.jsonl: the tries that were played again;
- soak.log: one line per run; heartbeat: touched on every message (soak.sh's freeze watchdog);
- done: written when every run has been played.

    python3 tools/trace/soak.py --report <out dir>      # counts and the runs that differ
"""
import gzip
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import playback as pb  # noqa: E402

# A run that doesn't match is played again from the start this many times; one that matches on a
# retry is recorded as "flaky" (the game sometimes misses a monster's hit at harness speed, with
# no engine difference behind it), one that fails every time keeps its status.
RETRIES = int(os.environ.get("SOAK_RETRIES", 1))
# Harness speed (frame time multiplier). 6 is the recording default; a lower one (SOAK_SPEED=2)
# checks whether a difference comes from the speed-up (a frame stall at 6 can let a death
# animation end the battle before a queued action, like Reaper's heal, has run).
# Game updates per drawn frame (HARNESS_TICKS, the harness's "ticks"): above 1, every update is a
# fixed 1/60 s times the speed and a frame runs that many before drawing once, so the game goes
# through the same states as at normal speed, faster (HarnessTicksPatch). Use speed 1 with it.
TICKS = int(os.environ.get("HARNESS_TICKS", 1))
# With ticks the speed is 1 unless SOAK_SPEED says otherwise: a larger one makes every tick a larger
# step again (speed 6 and ticks would be 0.1 s ticks), which is what ticks are there to avoid.
SPEED = float(os.environ.get("SOAK_SPEED", 1 if TICKS > 1 else 6))
# A run with no progress for this long is stuck: 3 minutes at the old speed; with ticks a step takes
# well under a second (30 s of real time is over an hour of game time), so 30 s (SOAK_STUCK).
STUCK_SECONDS = int(os.environ.get("SOAK_STUCK", 30 if TICKS > 1 else 180))
# The game applies some effects (an obtained card, and relics like Darkstone Periapt that react
# to it) on its effect list, which CommunicationMod doesn't wait for: when floor or HP differ,
# let the game run this many times 30 frames before calling it a difference.
SETTLE_TRIES = 5


def load_batch(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)["runs"]


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# Parallel soaks (soak.sh with SOAK_INSTANCES > 1): game copy SOAK_SHARD of SOAK_SHARDS plays every
# SOAK_SHARDS-th run of the batch, into <out dir>/shard-<SOAK_SHARD>.
SHARD = int(os.environ.get("SOAK_SHARD", 0))
SHARDS = int(os.environ.get("SOAK_SHARDS", 1))


def summary_rows(out):
    """The summary of a soak, its shards' included, by batch index."""
    paths = [os.path.join(out, "summary.jsonl")] + sorted(
        os.path.join(out, d, "summary.jsonl") for d in os.listdir(out) if d.startswith("shard-")) if os.path.isdir(out) else []
    return sorted((r for p in paths for r in read_jsonl(p)), key=lambda r: r["index"])


def report(out):
    rows = summary_rows(out)
    counts = Counter(r["status"] for r in rows)
    print(f"{len(rows)} runs played: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    for r in rows:
        if r["status"] != "matched":
            print(f"  #{r['index']:4} seed {r['seed']:>8} {r['status']:15} step {r['step']}/{r['steps']} "
                  f"floor {r.get('game_floor')} - {r.get('detail', '')}")
            print(f"        log: {r.get('log')}")


class Soak:
    def __init__(self, batch, out, delay):
        self.runs = load_batch(batch)
        self.out = out
        self.delay = delay
        os.makedirs(os.path.join(out, "logs"), exist_ok=True)
        self.summary = os.path.join(out, "summary.jsonl")
        self.started = os.path.join(out, "started.jsonl")
        self.tries = os.path.join(out, "tries.jsonl")
        self.heartbeat = os.path.join(out, "heartbeat")
        self.main_log = open(os.path.join(out, "soak.log"), "a")
        self.run_log = None
        self.speed_set = False
        self.phase = "idle"  # idle -> play -> leaving -> idle
        self.index = None

    # Logging ---------------------------------------------------------------------------
    def note(self, *a):
        print(time.strftime("%H:%M:%S"), *a, file=self.main_log, flush=True)

    def log(self, *a):
        if self.run_log:
            print(time.strftime("%H:%M:%S"), *a, file=self.run_log, flush=True)

    def send(self, cmd):
        self.log(">>", cmd)
        print(cmd, flush=True)

    # Run bookkeeping ---------------------------------------------------------------------
    def next_index(self):
        done = {r["index"] for r in read_jsonl(self.summary)}
        starts = Counter(r["index"] for r in read_jsonl(self.started))
        tries = Counter(r["index"] for r in read_jsonl(self.tries))
        for i in range(len(self.runs)):
            if i in done or i % SHARDS != SHARD:
                continue
            if self.runs[i].get("engine_changed"):
                # sts-cli commands: the engine can't take this run's decisions any more.
                self.write_summary(i, "engine_changed", 0, None, self.runs[i]["engine_changed"])
                continue
            if starts[i] - tries[i] >= 2:
                # The game died twice during this run (soak.sh restarted it): record and skip.
                self.write_summary(i, "crashed", 0, None, "the game stopped twice while playing this run")
                continue
            return i
        return None

    def write_summary(self, index, status, step, gs, detail="", log_path=None):
        run = self.runs[index]
        row = {
            "index": index, "seed": run["log"]["seed_string"], "status": status, "step": step,
            "steps": len(run["log"]["steps"]), "game_floor": gs and gs.get("floor"), "game_hp": gs and gs.get("current_hp"),
            "engine": run["engine"], "detail": detail, "log": log_path, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.summary, "a") as f:
            f.write(json.dumps(row) + "\n")

    def start_run(self):
        self.index = self.next_index()
        if self.index is None:
            self.note("all runs played")
            open(os.path.join(self.out, "done"), "w").close()
            self.phase = "finished"
            return
        run = self.runs[self.index]["log"]
        with open(self.started, "a") as f:
            f.write(json.dumps({"index": self.index, "time": time.strftime("%H:%M:%S")}) + "\n")
        current = os.path.join(self.out, "logs", "current.log")
        if os.path.exists(current) and os.path.getsize(current):
            # The run before was cut short (the game froze or died and soak.sh restarted it):
            # keep its log, named after the run it was playing.
            with open(current) as f:
                first = f.readline().split()
            tag = f"{int(first[2]):04d}-{first[4]}" if len(first) > 4 and first[1] == "run" else "unknown"
            os.replace(current, os.path.join(self.out, "logs", f"{tag}-interrupted-{time.strftime('%H%M%S')}.log"))
        self.run_log = open(current, "w")
        self.steps = run["steps"]
        self.i = 0
        self.pending = []
        self.choose_screen = None
        self.last = None
        self.errors = 0
        self.settle = 0
        self.same = pb.SameScreen()
        self.last_gs = None
        self.progress_t = time.time()
        self.phase = "play"
        self.log("run", self.index, "seed", run["seed_string"], "engine", self.runs[self.index]["engine"], len(self.steps), "steps")
        self.send(f"start {run.get('character', 'ironclad')} {run['ascension']} {run['seed_string']}")

    def finish(self, status, gs, detail=""):
        """Ends the current run: records it, keeps its log unless it matched, and leaves the game."""
        run = self.runs[self.index]["log"]
        self.log("RESULT", status, detail)
        if gs:
            pb.log_state(gs, self.log)
        self.run_log.close()
        self.run_log = None
        current = os.path.join(self.out, "logs", "current.log")
        earlier = [r for r in read_jsonl(self.tries) if r["index"] == self.index]
        log_path = None
        if status == "matched" and not earlier:
            os.remove(current)
        else:
            try_tag = f"-try{len(earlier) + 1}" if earlier or (status != "matched" and RETRIES) else ""
            log_path = os.path.join(self.out, "logs", f"{self.index:04d}-{run['seed_string']}-{status}{try_tag}.log")
            os.replace(current, log_path)
        self.note(f"#{self.index} {run['seed_string']} {status} at step {self.i}/{len(self.steps)} {detail}")
        if status != "matched" and len(earlier) < RETRIES:
            # Play it again before calling it a difference.
            with open(self.tries, "a") as f:
                f.write(json.dumps({"index": self.index, "status": status, "step": self.i, "detail": detail, "log": log_path}) + "\n")
            self.note(f"#{self.index} playing it again")
        elif status == "matched" and earlier:
            first = earlier[0]
            self.write_summary(self.index, "flaky", first["step"], gs,
                               f"matched on try {len(earlier) + 1}; try 1 {first['status']}: {first['detail']}", first["log"])
        else:
            if earlier:
                detail = f"{detail} (every one of {len(earlier) + 1} tries)"
            self.write_summary(self.index, status, self.i, gs, detail, log_path)
        self.phase = "leaving"
        if gs is not None and gs.get("screen_type") == "GAME_OVER":
            print("proceed", flush=True)
        else:
            print('harness {"abandon": true}', flush=True)

    # The message loop --------------------------------------------------------------------
    def handle(self, msg):
        os.utime(self.heartbeat) if os.path.exists(self.heartbeat) else open(self.heartbeat, "w").close()
        avail = msg.get("available_commands", [])
        in_game = msg.get("in_game")
        gs = msg.get("game_state") if in_game else None

        if self.phase in ("idle", "finished"):
            if not in_game and self.phase == "idle":
                self.start_run()
            elif in_game and self.phase == "idle":
                # The game came up inside a run (e.g. after a restart): leave it first.
                self.phase = "leaving"
                print("proceed" if gs.get("screen_type") == "GAME_OVER" else 'harness {"abandon": true}', flush=True)
            return
        if self.phase == "leaving":
            if not in_game:
                self.phase = "idle"
                self.start_run()
            elif gs.get("screen_type") == "GAME_OVER" and "proceed" in avail:
                print("proceed", flush=True)
            else:
                print("wait 30", flush=True)
            return

        # Playing a run.
        if "error" in msg:
            self.log("error", msg["error"])
            self.errors += 1
            if self.errors >= 5:
                return self.finish("refused", self.last_gs, f"the game keeps refusing {self.last!r}")
            if self.last:
                self.pending.insert(0, self.last)
            self.last = None
            # An error reply carries no game state, and the game sends nothing more until it gets
            # a command: ask for the state, and carry on from that.
            self.send("state")
            return
        if not in_game:
            return  # still starting the run
        self.last_gs = gs
        if not self.speed_set:
            self.speed_set = True
            self.send(f'harness {{"speed": {SPEED}, "ticks": {TICKS}}}')
            return
        if time.time() - self.progress_t > STUCK_SECONDS:
            return self.finish("stuck", gs, f"no progress for {STUCK_SECONDS} s on screen {gs.get('screen_type')}, available {avail}")

        engine = self.runs[self.index]["engine"]
        if gs.get("screen_type") == "GAME_OVER":
            victory = bool(gs.get("screen_state", {}).get("victory"))
            if self.i < len(self.steps):
                return self.finish("ended_early", gs, f"game over (victory {victory}) with {len(self.steps) - self.i} decisions left; engine {engine}")
            if victory != bool(engine["won"]):
                return self.finish("result_differs", gs, f"game victory {victory}, engine won {engine['won']}")
            return self.finish("matched", gs)

        if self.pending:
            cmd = self.pending.pop(0)
            optional = cmd.endswith("?")
            cmd = cmd.rstrip("?")
            verb = cmd.split()[0]
            # An optional confirm belongs to the screen its picks went to (with no picks, the one
            # its step began on: run GSWN, Purity exhausting nothing): if another screen has
            # opened since (the next selection, fast at harness ticks: run GBFK, Toolbox's pick
            # then Gambling Chip's; even one of the same type, pb.screen_id: run H134, Meditate's
            # grid then Foresight's Scry), it would close that one with nothing picked.
            if optional and (verb not in avail or (verb == "confirm" and pb.screen_id(gs) != self.choose_screen)):
                self.send("wait 1")
                return
            if verb == "choose":
                self.choose_screen = pb.screen_id(gs)
            if verb not in avail and verb != "wait":
                self.pending.insert(0, cmd)
                self.send("wait 30")
                return
            time.sleep(self.delay)
            self.last = cmd
            self.same.sent(gs)
            self.send(cmd)
            return
        self.last = None
        if self.i >= len(self.steps):
            # The engine's run is over; click through to the game's end screen.
            self.send(pb.click_through(gs, avail, "over"))
            return
        step = self.steps[self.i]
        if pb.screen_kind(gs, step["kind"]) != step["kind"]:
            if not any(c in avail for c in ("play", "end", "choose", "proceed", "confirm")):
                self.send("wait 30")
            elif self.same.stale(gs):
                self.log("the screen the last command was sent on: waiting for the game")
                self.send("wait 30")
            else:
                cmd = pb.click_through(gs, avail, step["kind"])
                if not cmd.startswith("wait"):
                    self.same.sent(gs)
                    # What was clicked through, to see a click that landed on the wrong screen.
                    self.log("click-through on", gs.get("screen_type"), "floor", gs.get("floor"), "choices", gs.get("choice_list"))
                self.send(cmd)
            return
        hp = gs.get("current_hp")
        if step.get("hp") is not None and (gs.get("floor"), hp) != (step.get("floor"), step["hp"]):
            if self.settle < SETTLE_TRIES:
                self.settle += 1
                self.log("floor", gs.get("floor"), "hp", hp, "differ from the engine's", step.get("floor"), step["hp"],
                         "- letting the game settle", self.settle)
                self.send("wait 30")
                return
            return self.finish("differs", gs, f"before decision {self.i} ({step.get('action')}): game floor {gs.get('floor')} hp {hp}, "
                                              f"engine floor {step.get('floor')} hp {step['hp']}")
        pending = pb.resolve(step, gs)
        if pending is None:
            # The game isn't ready for this step's commands yet (see pb.resolve).
            self.send("wait 30")
            return
        self.log("step", self.i, "floor", gs.get("floor"), "hp", hp, "gold", gs.get("gold"), step.get("action"))
        pb.log_state(gs, self.log)
        self.pending = pending
        self.choose_screen = pb.screen_id(gs)
        self.i += 1
        self.progress_t = time.time()
        self.errors = 0
        self.settle = 0
        self.send("wait 1")


def main():
    if sys.argv[1] == "--report":
        return report(sys.argv[2])
    out = sys.argv[2] if SHARDS == 1 else os.path.join(sys.argv[2], f"shard-{SHARD}")
    soak = Soak(sys.argv[1], out, float(sys.argv[3]) if len(sys.argv) > 3 else 0.1)
    soak.note("soak starting:", len(soak.runs), "runs from", sys.argv[1], f"(copy {SHARD} of {SHARDS})" if SHARDS > 1 else "")
    print("ready", flush=True)
    try:
        for line in sys.stdin:
            soak.handle(json.loads(line))
    except Exception as e:  # stdout belongs to the game
        soak.note("crash", repr(e))
        soak.log("crash", repr(e))
        raise


if __name__ == "__main__":
    main()
