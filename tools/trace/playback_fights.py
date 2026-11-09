"""Watch fights played by the network (python/watch/watch_fights.py) in the real game.

Also practice fights (python/watch/watch_practice.py): each carries its character, ascension and
the harness commands that set it up (its relics, then its deck, HP and potions and the
encounter), which this sends in turn.

Set this script as CommunicationMod's command, e.g.

    command=/opt/homebrew/bin/python3 -I <repo>/tools/trace/playback_fights.py <fights.json> [delay seconds]

and launch the modded game (see tools/trace/setup.sh). For each fight it starts an Ironclad run
on the fight's seed, sets the deck with the `harness` command and starts the encounter on floor 1
(the same fight the engine played), then sends the saved moves one by one. After each fight it
abandons the run and starts the next one.

Stdout is CommunicationMod's command channel; progress goes to <fights.json>.playback.log, with
a warning whenever the game's HP differs from the engine's at a move.
"""
import json
import sys
import time

LOG_PATH = sys.argv[1] + ".playback.log"
FIGHTS = json.load(open(sys.argv[1]))["fights"]
DELAY = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
LOG = open(LOG_PATH, "a")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=LOG, flush=True)


def send(cmd):
    log(">>", cmd)
    print(cmd, flush=True)


def recv():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return json.loads(line)


def main():
    print("ready", flush=True)
    log("playing", len(FIGHTS), "fights")
    f = 0
    phase = "start"  # start -> setup -> fight -> done (fight over) -> start (next fight)
    moves, pending, last = [], [], None
    setup = []  # the fight's harness commands still to send
    started = False  # the fight's combat has begun (until then the game is still on Neow)
    while True:
        msg = recv()
        if "error" in msg:
            # The game was busy when the command arrived; send it again.
            log("error", msg["error"])
            if last:
                pending.insert(0, last)
            last = None
        avail = msg.get("available_commands", [])
        if not msg.get("in_game"):
            if f >= len(FIGHTS):
                log("all fights played")
                while True:
                    recv()
            if phase in ("start", "done"):
                fight = FIGHTS[f]
                where = f"Act {fight['act']} ({fight['practice']} deck), " if "act" in fight else ""
                log(f"fight {f + 1}: {where}{fight['encounter']}, seed {fight['seed_string']}, engine result "
                    f"{'won' if fight['won'] else 'lost'} with {fight['hp']} HP")
                send(f"start {fight.get('character', 'ironclad')} {fight.get('ascension', 0)} {fight['seed_string']}")
                setup = [dict(c) for c in fight["harness"]] if "harness" in fight else [
                    {"deck": fight["deck"], "hp": 80, "max_hp": 80, "encounter": fight["encounter"]}]
                setup[0]["speed"] = 1
                phase = "setup"
            continue
        gs = msg["game_state"]
        screen = gs.get("screen_type")
        in_combat = gs.get("room_phase") == "COMBAT" and "combat_state" in gs
        if phase == "setup" and not in_combat:
            if screen == "GRID" and "confirm" in avail:
                # Pandora's Box shows the cards it transformed into.
                send("confirm")
                continue
            if len(setup) < len(FIGHTS[f].get("harness", [None])):
                # Relics given: their cards (Pandora's Box's, Necronomicon's curse) reach the deck
                # once their animations end; the deck command replaces the deck after that.
                time.sleep(4)
            send("harness " + json.dumps(setup.pop(0)))
            if not setup:
                moves, pending, started = list(FIGHTS[f]["moves"]), [], False
                phase = "fight"
            continue
        if phase == "fight" and in_combat:
            started = True
        if phase == "fight" and not started:
            send("wait 30")
            continue
        if phase == "fight" and (screen == "GAME_OVER" or not in_combat):
            log(f"fight {f + 1} over: game HP {gs.get('current_hp')}, {len(moves)} saved moves left")
            f += 1
            phase = "done"
            time.sleep(3)
            if screen == "GAME_OVER":
                send("proceed")
            else:
                send('harness {"abandon": true}')
            continue
        if phase == "done":
            send("proceed" if screen == "GAME_OVER" and "proceed" in avail else "wait 30")
            continue
        if phase != "fight":
            send("wait 30")
            continue
        # In the fight: send the next command once the game takes it.
        if not pending:
            if "play" not in avail and "end" not in avail and "choose" not in avail and "confirm" not in avail:
                send("wait 30")
                continue
            if not moves:
                log("saved moves ran out with the fight still going")
                send("wait 30")
                continue
            m = moves.pop(0)
            hp = gs.get("current_hp")
            if hp != m["hp"]:
                log(f"WARNING: game HP {hp}, engine HP {m['hp']} before {m['cmds']} (turn {m['turn']})")
            pending = list(m["cmds"])
        cmd = pending.pop(0)
        optional = cmd.endswith("?")
        cmd = cmd.rstrip("?")
        verb = cmd.split()[0]
        if verb not in avail:
            if not optional:
                pending.insert(0, cmd)
            send("wait 30")
            continue
        time.sleep(DELAY)
        last = cmd
        send(cmd)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # stdout belongs to the game
        log("crash", repr(e))
        raise
