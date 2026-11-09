"""Watch a run played in the CLI (`sts-cli run --save run.json`) in the real game.

Set this script as CommunicationMod's command, e.g.

    command=/opt/homebrew/bin/python3 -I <repo>/tools/trace/playback.py <run.json> [delay seconds]

and launch the modded game (see tools/trace/setup.sh). It starts an Ironclad run on the saved
seed and ascension, then sends the saved choices one by one. Screens the CLI skips over (Neow's
"Talk", opening a chest, leaving a shop or rest site) are clicked through automatically. When
the saved steps run out the game is left where the CLI run stopped.

Stdout is CommunicationMod's command channel; progress goes to <run.json>.playback.log.
tools/trace/soak.py reuses the screen handling here to play back many runs unattended.
"""
import json
import sys
import time

RUN = None
DELAY = 0.4
LOG = sys.stderr


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


def screen_kind(gs, want=None):
    """The CLI screen kind this game state corresponds to (None if it is a click-through), when the
    next saved step is of kind `want`."""
    screen = gs.get("screen_type")
    ss = gs.get("screen_state", {})
    if screen == "REST" and not gs.get("choice_list") and want == "rest":
        # A rest site with no option to use (Coffee Dripper, Fusion Hammer, the Ruby key taken): the
        # game completes the room at once, while the engine stops there with only `proceed` (soak
        # 2026-10-27, run GJQN: the playback clicked through and waited on the map).
        return "rest"
    if gs.get("room_phase") == "COMBAT" and "combat_state" in gs and screen in ("NONE", "HAND_SELECT", "GRID", "CARD_REWARD"):
        return "combat"
    if screen == "EVENT":
        opts = ss.get("options", [])
        if ss.get("event_id") == "Neow Event":
            return "neow" if len(opts) >= 2 else None
        return "event"
    return {
        "MAP": "map",
        "COMBAT_REWARD": "rewards",
        "CARD_REWARD": "card_reward",
        "GRID": "grid",
        "BOSS_REWARD": "boss_relic",
        "SHOP_SCREEN": "shop",
        "REST": "rest" if gs.get("choice_list") else None,
    }.get(screen)


def click_through(gs, avail, want):
    """A command that moves past a screen the CLI does not stop on."""
    screen = gs.get("screen_type")
    if screen == "EVENT":
        return "choose 0"
    if screen == "CHEST":
        return "choose 0" if gs.get("choice_list") else "proceed"
    if screen == "SHOP_ROOM":
        return "choose 0" if want == "shop" and "choose" in avail else "proceed"
    if screen == "SHOP_SCREEN":
        return "leave"
    if screen in ("GRID", "HAND_SELECT") and "confirm" in avail:
        return "confirm"
    for c in ("proceed", "confirm", "skip"):
        if c in avail:
            return c
    return "wait 30"


class SameScreen:
    """Notices a state showing exactly the screen the last command was sent on. CommunicationMod
    also reports changes nobody asked for (the gold from Ceramic Fish when an event's card finishes
    being obtained), and such a message can come before the game has acted on the command; clicking
    through that stale screen clicks on whatever comes next (an event's "leave" turned into a map
    node). Waiting for the next state is safe: up to `limit` times, then it clicks through anyway."""

    def __init__(self, limit=5):
        self.acted_on = None
        self.waits = 0
        self.limit = limit

    @staticmethod
    def signature(gs):
        return json.dumps({k: gs.get(k) for k in ("screen_type", "screen_state", "choice_list", "floor", "act")}, sort_keys=True)

    def sent(self, gs):
        self.acted_on = self.signature(gs) if gs else None
        self.waits = 0

    def stale(self, gs):
        if self.acted_on is not None and self.signature(gs) == self.acted_on and self.waits < self.limit:
            self.waits += 1
            return True
        return False


def resolve(step, gs):
    """The game commands for one saved step, given the game's current state; None while the game
    isn't ready for them yet (a deck grid whose cards haven't come through)."""
    if "path_x" in step:
        x = step["path_x"]
        nodes = gs.get("screen_state", {}).get("next_nodes", [])
        for i, n in enumerate(nodes):
            if n["x"] == x:
                return [f"choose {i}"]
        return ["choose boss"]
    if "grid_deck" in step:
        deck = gs.get("deck", [])
        cards = gs.get("screen_state", {}).get("cards", [])
        out = []
        for d in step["grid_deck"]:
            uuid = deck[d]["uuid"] if d < len(deck) else None
            k = next((i for i, c in enumerate(cards) if c["uuid"] == uuid), None)
            if k is None:
                # Right after opening, the grid's cards can still be missing or stale: picking card 0
                # then took the wrong card (2026-10-17 soak, a shop removal). Wait and look again.
                return None
            out.append(f"choose {k}")
        return out + ["confirm?"]
    return list(step.get("cmds", []))


def screen_id(gs):
    """A screen as its type and, for a selection, the cards it offers, picked or not: the next grid
    or hand selection, of the same type, counts as another screen (run H134 of the 2026-10-28 Watcher
    soak: Meditate's two picks closed its grid, the next turn's Foresight Scry opened another, and
    the optional confirm meant for Meditate's closed the Scry with nothing picked)."""
    st = gs.get("screen_state") or {}
    cards = [c for k in ("cards", "selected_cards", "hand", "selected") for c in (st.get(k) or []) if isinstance(c, dict)]
    # A set: a grid's picks stay among its cards as well (a hand selection's move out of the hand).
    return gs.get("screen_type"), tuple(sorted({str(c.get("uuid")) for c in cards}))


def card_names(cards):
    return [c["name"] for c in cards]  # CommunicationMod's names already carry the "+"


def log_state(gs, log=log):
    """The game's hand (in combat) or deck, to compare with the CLI when they drift apart."""
    cs = gs.get("combat_state")
    if gs.get("room_phase") == "COMBAT" and cs:
        p = cs.get("player", {})
        log("   game hand", card_names(cs.get("hand", [])), "energy", p.get("energy"))
        powers = lambda c: " ".join(f"{x.get('id')}:{x.get('amount')}" for x in c.get("powers", []))
        log("   game player block", p.get("block"), powers(p))
        for m in cs.get("monsters", []):
            log("   game monster", m.get("id"), m.get("current_hp"), "block", m.get("block"), m.get("intent"),
                m.get("move_adjusted_damage"), "x", m.get("move_hits"), "gone" if m.get("is_gone") else "", powers(m))
        # The draw pile in order, top first (the game's group ends with the top card): a draw that
        # differs shows where the order did (run H141 of the 2026-10-28 Watcher soak).
        log("   game draw pile", card_names(list(reversed(cs.get("draw_pile", [])))))
    else:
        log("   game deck", card_names(gs.get("deck", [])))
        # The relics too: a different relic (a shop slot taken wrongly) shows only much later.
        log("   game relics", [r.get("id") for r in gs.get("relics", [])])
        if gs.get("screen_type") == "GRID":
            log("   game grid", card_names(gs.get("screen_state", {}).get("cards", [])))
        if gs.get("screen_type") == "EVENT":
            ss = gs.get("screen_state", {})
            log("   game event", ss.get("event_id"), "choices", gs.get("choice_list"),
                "options", [(o.get("label"), o.get("disabled")) for o in ss.get("options", [])])
        if gs.get("screen_type") == "SHOP_SCREEN":
            ss = gs.get("screen_state", {})
            log("   game shop relics", [(r.get("id"), r.get("price")) for r in ss.get("relics", [])],
                "cards", [(c.get("name"), c.get("price")) for c in ss.get("cards", [])],
                "purge", ss.get("purge_cost") if ss.get("purge_available") else None)
    # Relic counters (Sundial's shuffles, Nunchaku's attacks...): a counter that differs fires a
    # relic early or late (runs H0YY and H0XC of the 2026-10-28 Watcher soak).
    counters = [f"{r.get('id')}={r.get('counter')}" for r in gs.get("relics", []) if r.get("counter") is not None and r["counter"] >= 0]
    if counters:
        log("   game relic counters", " ".join(counters))
    # The run's RNG counters (the harness's state), to line up with the engine's when they differ.
    h = gs.get("harness") or {}
    if h.get("rng"):
        names = ("card", "misc", "merchant", "potion", "relic", "event", "treasure", "monster")
        log("   game rng", " ".join(f"{k}={h['rng'][k].get('counter')}" for k in names if k in h["rng"]), "card_blizz", h.get("card_blizz"))


def main():
    print("ready", flush=True)
    steps = RUN["steps"]
    log("playing", len(steps), "steps of seed", RUN["seed_string"])
    i = 0
    pending = []
    choose_screen = None
    last = None
    errors = 0
    last_gs = None
    same = SameScreen()
    while True:
        msg = recv()
        if "error" in msg:
            # The game was busy when the command arrived; send it again. A command the game
            # keeps refusing means the game and the CLI have drifted apart: stop there.
            log("error", msg["error"])
            errors += 1
            if errors >= 5:
                log("the game keeps refusing", last, "- stopping; the game and the CLI differ here")
                if last_gs:
                    log_state(last_gs)
                while True:
                    recv()
            if last:
                pending.insert(0, last)
            last = None
        else:
            errors = 0
        avail = msg.get("available_commands", [])
        if not msg.get("in_game"):
            if i == 0 and not pending:
                send(f"start {RUN.get('character', 'ironclad')} {RUN['ascension']} {RUN['seed_string']}")
            continue
        gs = msg["game_state"]
        last_gs = gs
        if gs.get("screen_type") == "GAME_OVER":
            log("game over")
            while True:
                recv()
        if pending:
            cmd = pending.pop(0)
            optional = cmd.endswith("?")
            cmd = cmd.rstrip("?")
            verb = cmd.split()[0]
            # An optional confirm belongs to the screen its picks went to (with no picks, the one
            # its step began on): if another screen has opened since (the next selection, fast at
            # harness ticks, even one of the same type: screen_id), it would close that one.
            if optional and (verb not in avail or (verb == "confirm" and screen_id(gs) != choose_screen)):
                send("wait 1")
                continue
            if verb == "choose":
                choose_screen = screen_id(gs)
            if verb not in avail and verb != "wait":
                pending.insert(0, cmd)
                send("wait 30")
                continue
            time.sleep(DELAY)
            last = cmd
            same.sent(gs)
            send(cmd)
            continue
        last = None
        if i >= len(steps):
            log("all saved steps played")
            while True:
                recv()
        want = steps[i]["kind"]
        have = screen_kind(gs, want)
        if have != want:
            if "play" not in avail and "end" not in avail and "choose" not in avail and "proceed" not in avail and "confirm" not in avail:
                send("wait 30")
            elif same.stale(gs):
                log("the screen the last command was sent on: waiting for the game")
                send("wait 30")
            else:
                cmd = click_through(gs, avail, want)
                if not cmd.startswith("wait"):
                    same.sent(gs)
                send(cmd)
            continue
        pending = resolve(steps[i], gs)
        if pending is None:
            send("wait 30")
            continue
        hp = gs.get("current_hp")
        if steps[i].get("hp") is not None and (gs.get("floor"), hp) != (steps[i].get("floor"), steps[i]["hp"]):
            log("WARNING: game and CLI differ here: game floor", gs.get("floor"), "hp", hp, "CLI floor", steps[i].get("floor"), "hp", steps[i]["hp"])
        log("step", i, "floor", gs.get("floor"), "hp", hp, "gold", gs.get("gold"), steps[i].get("action"))
        log_state(gs)
        choose_screen = screen_id(gs)
        i += 1
        if not pending:
            send("wait 1")
        else:
            pending.insert(0, pending.pop(0))
            send("wait 1")


if __name__ == "__main__":
    RUN = json.load(open(sys.argv[1]))
    DELAY = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    LOG = open(sys.argv[1] + ".playback.log", "a")
    try:
        main()
    except Exception as e:  # stdout belongs to the game
        log("crash", repr(e))
        raise
