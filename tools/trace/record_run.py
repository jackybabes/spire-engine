"""Record whole Act 1 runs from the real game through CommunicationMod.

Like record.py, but instead of setting up single fights it starts seeded Ironclad runs and plays
them with a random policy restricted to what the port covers so far:
  - Neow blessings that give no colorless card (colorless cards are not ported yet);
  - any map path; events get random options (Match and Keep is abandoned);
  - shops: random purchases of Ironclad cards, relics and potions, and card removal;
  - takes most relics; drinks potions now and then in fights, except the card-discovery ones,
    whose RNG use in the game depends on frame timing (those are discarded on the map).

The run is abandoned at the boss relic screen of act <max act> (default 1); earlier acts take a
random boss relic and carry on.

Every state and command goes to <out>/run-<n>.jsonl, which `sts-cli replay-run` checks.

Usage (as CommunicationMod's command):
    python3 -I record_run.py <out dir> <runs> <policy seed> [speed] [max act] [--character silent|defect|watcher]
        [--relics A,B]
With --character silent, defect or watcher the runs are that character's; she buys any card in shops
(colorless ones are ported now).
"""
import json
import os
import random
import sys
import time
from pathlib import Path

CHARACTER = "ironclad"
if "--character" in sys.argv:
    i = sys.argv.index("--character")
    CHARACTER = sys.argv[i + 1].lower()
    del sys.argv[i:i + 2]
# --relics A,B: relics (IDs, spaces as `_`) added after Neow with the HP override, for relics
# whose effects only show in runs (rewards, shops, gold). The replay adds them too.
FOCUS_RELICS = []
if "--relics" in sys.argv:
    i = sys.argv.index("--relics")
    FOCUS_RELICS = [r.replace("_", " ") for r in sys.argv[i + 1].split(",")]
    del sys.argv[i:i + 2]
OUT = Path(sys.argv[1])
RUNS = int(sys.argv[2])
RNG = random.Random(int(sys.argv[3]))
SPEED = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
# Game updates per drawn frame (HARNESS_TICKS, the harness's "ticks"): above 1, every update is a
# fixed 1/60 s times the speed and a frame runs that many before drawing once, so the game goes
# through the same states as at normal speed, faster (HarnessTicksPatch). Use speed 1 with it.
TICKS = int(os.environ.get("HARNESS_TICKS", 1))
MAX_ACT = int(sys.argv[5]) if len(sys.argv) > 5 else 1
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "recorder.log", "a")
HP = 400
SEED_CHARS = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
# Cards whose in-combat behaviour is not ported (the combat replay would stop at them).
AVOID_CARDS = set()
NEOW_AVOID = ("colorless",)
# Ironclad cards (the shop also sells colorless ones, which are not ported yet).
RED = """Anger Armaments Barricade Battle_Trance Berserk Blood_for_Blood Bloodletting Bludgeon
Body_Slam Brutality Burning_Pact Carnage Clash Cleave Clothesline Combust Corruption Dark_Embrace
Demon_Form Disarm Double_Tap Dropkick Dual_Wield Entrench Evolve Exhume Feed Feel_No_Pain Fiend_Fire
Fire_Breathing Flame_Barrier Flex Ghostly_Armor Havoc Headbutt Heavy_Blade Hemokinesis Immolate
Impervious Inflame Intimidate Iron_Wave Juggernaut Limit_Break Metallicize Offering Perfected_Strike
Pommel_Strike Power_Through Pummel Rage Rampage Reaper Reckless_Charge Rupture Searing_Blow
Second_Wind Seeing_Red Sentinel Sever_Soul Shockwave Shrug_It_Off Spot_Weakness Sword_Boomerang
Thunderclap True_Grit Twin_Strike Uppercut Warcry Whirlwind Wild_Strike""".split()
RED = [c.replace("_", " ") for c in RED]
RED_SET = set(RED) | {"Infernal Blade"}
DISCOVERY_POTIONS = {"AttackPotion", "SkillPotion", "PowerPotion", "ColorlessPotion"}


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


def safe_nodes(gs):
    """Map nodes from which the boss can be reached without entering a ? or $ room."""
    nodes = {(n["x"], n["y"]): n for n in gs.get("map", [])}
    safe = set()
    for y in range(14, -1, -1):
        for (x, ny), n in nodes.items():
            if ny != y or n["symbol"] == "?":
                continue
            kids = [(c["x"], c["y"]) for c in n["children"]]
            if y == 14 or any(k in safe for k in kids):
                safe.add((x, ny))
    return safe


def combat_options(gs):
    cs = gs["combat_state"]
    opts = []
    for i, c in enumerate(cs["hand"]):
        if not c["is_playable"]:
            continue
        if c["has_target"]:
            for t, m in enumerate(cs["monsters"]):
                if not m["is_gone"] and not m.get("half_dead"):
                    opts.append(f"play {i + 1} {t}")
        else:
            opts.append(f"play {i + 1}")
    opts.append("end")
    return opts


class Recorder:
    def __init__(self):
        self.run = 0
        self.out = None
        self.cards_opened = 0  # card rewards opened on the current reward screen
        self.skipped = set()  # relic rewards left on the current reward screen
        self.shopped = False  # opened the merchant in the current shop room
        self.sped_up = False
        self.buffed = False
        # A Smoke Bomb was used in this fight: the game still takes commands during the 2.5 s
        # escape, which the engine doesn't model (the fight ends there), so wait it out.
        self.smoked = False

    def write(self, obj):
        if self.out is not None:
            self.out.write(json.dumps(obj) + "\n")

    def finish(self, why):
        if self.out is not None:
            self.write({"type": "end", "why": why})
            self.out.close()
            self.out = None
            self.run += 1
            log("run finished", self.run, why)

    def abandon(self, why):
        self.finish(why)
        send('harness {"abandon": true}')

    def decide(self, msg):
        avail = msg.get("available_commands", [])
        gs = msg["game_state"]
        screen = gs.get("screen_type")
        ss = gs.get("screen_state", {})
        in_combat = gs.get("room_phase") == "COMBAT" and "combat_state" in gs

        if screen == "GAME_OVER":
            self.finish("game over")
            return "proceed"
        if not in_combat:
            self.smoked = False
        if in_combat:
            if self.smoked:
                return "wait 30"
            if screen in ("HAND_SELECT", "GRID") and "choose" not in avail and "confirm" in avail:
                return "confirm"
            if screen == "HAND_SELECT" and "confirm" in avail and (ss.get("selected") or (ss.get("can_pick_zero") and RNG.random() < 0.5)):
                # Pick at most one card from any-number selections (Gambling Chip), so each
                # recorded choice maps to one engine choice.
                return "confirm"
            if "choose" in avail and screen == "GRID":
                # Picking a selected card again would unselect it: pick among the others. A grid
                # of any number (Scry) stops at a random point.
                picked = {c.get("uuid") for c in ss.get("selected_cards", [])}
                free = [i for i, c in enumerate(ss.get("cards", [])) if c.get("uuid") not in picked]
                done = len(picked) >= ss.get("num_cards", 1)
                if "confirm" in avail and (not free or done or (ss.get("any_number") and RNG.random() < 0.35)):
                    return "confirm"
                return f"choose {RNG.choice(free)}" if free else "wait 30"
            if "choose" in avail and screen == "HAND_SELECT":
                return f"choose {RNG.randrange(len(gs['choice_list']))}"
            if "choose" in avail and screen == "CARD_REWARD":
                # A card choice inside the fight: Wish, Stance Potion, Foreign Influence.
                return f"choose {RNG.randrange(len(gs['choice_list']))}"
            usable = [(i, p) for i, p in enumerate(gs.get("potions", [])) if p.get("can_use") and p["id"] not in DISCOVERY_POTIONS]
            if "potion" in avail and "end" in avail and usable and RNG.random() < 0.08:
                i, p = RNG.choice(usable)
                self.smoked = p["id"] == "SmokeBomb"
                if p.get("requires_target"):
                    targets = [t for t, m in enumerate(gs["combat_state"]["monsters"]) if not m["is_gone"] and not m.get("half_dead")]
                    return f"potion use {i} {RNG.choice(targets)}"
                return f"potion use {i}"
            if "play" in avail or "end" in avail:
                # Mostly play cards (random play otherwise dies on floor 2 or 3).
                opts = combat_options(gs)
                plays = opts[:-1]
                if plays and RNG.random() < 0.9:
                    return RNG.choice(plays)
                return "end"
            if "confirm" in avail:
                return "confirm"
            return "wait 30"
        if screen == "MAP":
            self.shopped = False
        if screen != "COMBAT_REWARD" and screen != "CARD_REWARD":
            self.cards_opened = 0
            self.skipped = set()
        if screen == "EVENT":
            opts = [o for o in ss.get("options", []) if not o.get("disabled")]
            if ss.get("event_id") != "Neow Event" and ss.get("event_name") != "Neow":
                if ss.get("event_id") == "Match and Keep!":
                    return self.abandon("diverged: Match and Keep")
                return f"choose {RNG.choice(opts)['choice_index']}" if opts else "proceed"
            if len(opts) >= 2:
                ok = [o for o in opts if not any(k in o["text"].lower() for k in NEOW_AVOID)]
                pick = RNG.choice(ok or opts)
                return f"choose {pick['choice_index']}"
            return f"choose {opts[0]['choice_index']}" if opts else "proceed"
        if screen == "SHOP_ROOM":
            if not self.shopped and "choose" in avail:
                self.shopped = True
                return "choose 0"
            return "proceed"
        if screen == "SHOP_SCREEN":
            items = []
            if ss.get("purge_available") and gs["gold"] >= ss.get("purge_cost", 999):
                items.append("purge")
            items += [("card", c) for c in ss.get("cards", []) if c["price"] <= gs["gold"]]
            items += [("relic", r) for r in ss.get("relics", []) if r["price"] <= gs["gold"]]
            items += [("potion", p) for p in ss.get("potions", []) if p["price"] <= gs["gold"]]
            free = any(p["id"] == "Potion Slot" for p in gs.get("potions", []))
            ok = [i for i, it in enumerate(items)
                  if it == "purge" or (it[0] == "card" and (CHARACTER != "ironclad" or it[1]["name"].rstrip("+") in RED_SET))
                  or it[0] == "relic" or (it[0] == "potion" and free)]
            if ok and RNG.random() < 0.6:
                return f"choose {RNG.choice(ok)}"
            return "leave" if "leave" in avail else ("return" if "return" in avail else "proceed")
        if screen == "MAP" and not self.buffed:
            # Random play dies early; a big HP pool lets runs reach the boss. The replay applies
            # the same change to the engine.
            self.buffed = True
            if FOCUS_RELICS:
                held = [r["id"] for r in gs.get("relics", [])]
                relics = held + [r for r in FOCUS_RELICS if r not in held]
                return "harness " + json.dumps({"max_hp": HP, "hp": HP, "relics": relics})
            return 'harness {"max_hp": %d, "hp": %d}' % (HP, HP)
        discard = [i for i, p in enumerate(gs.get("potions", [])) if p["id"] in DISCOVERY_POTIONS]
        if screen == "MAP" and discard and "potion" in avail:
            return f"potion discard {discard[0]}"
        if screen == "MAP":
            nodes = ss.get("next_nodes", [])
            if ss.get("boss_available") and not nodes:
                return "choose boss"
            return f"choose {RNG.randrange(len(nodes))}"
        if screen == "COMBAT_REWARD":
            for i, r in enumerate(ss.get("rewards", [])):
                t = r["reward_type"]
                if t in ("GOLD", "STOLEN_GOLD"):
                    return f"choose {i}"
                if t == "POTION" and any(p["id"] == "Potion Slot" for p in gs.get("potions", [])) and RNG.random() < 0.5:
                    return f"choose {i}"
                if t == "RELIC" and RNG.random() < 0.85 and (i, t) not in self.skipped:
                    return f"choose {i}"
                if t == "RELIC":
                    self.skipped.add((i, t))
                if t in ("EMERALD_KEY", "SAPPHIRE_KEY") and RNG.random() < 0.6 and (i, t) not in self.skipped:
                    return f"choose {i}"
                if t in ("EMERALD_KEY", "SAPPHIRE_KEY"):
                    self.skipped.add((i, t))
                if t == "CARD" and self.cards_opened == 0:
                    self.cards_opened += 1
                    return f"choose {i}"
            return "proceed"
        if screen == "CARD_REWARD":
            cards = [i for i, c in enumerate(ss.get("cards", [])) if c["name"].rstrip("+") not in AVOID_CARDS]
            if "bowl" in gs.get("choice_list", []) and RNG.random() < 0.2:
                return "choose bowl"
            if cards and RNG.random() < 0.8:
                return f"choose {RNG.choice(cards)}"
            return "skip" if "skip" in avail else "proceed"
        if screen == "GRID":
            if "confirm" in avail and (ss.get("confirm_up") or len(ss.get("selected_cards", [])) >= ss.get("num_cards", 1)):
                return "confirm"
            chosen = {c["uuid"] for c in ss.get("selected_cards", [])}
            options = [i for i, c in enumerate(ss.get("cards", [])) if c["uuid"] not in chosen]
            return f"choose {RNG.choice(options)}"
        if screen == "REST":
            opts = gs.get("choice_list", [])
            if not opts:
                return "proceed"
            if "rest" in opts and gs["current_hp"] * 2 < gs["max_hp"]:
                return "choose rest"
            if "recall" in opts and RNG.random() < 0.4:
                return "choose recall"
            known = [o for o in opts if o in ("rest", "smith", "lift", "toke", "dig")]
            return f"choose {RNG.choice(known)}" if known else "proceed"
        if screen == "CHEST":
            return "choose 0" if gs.get("choice_list") else "proceed"
        if screen == "BOSS_REWARD":
            if gs.get("act", 1) >= MAX_ACT:
                return self.abandon("act cleared")
            relics = ss.get("relics", [])
            if relics and "choose" in avail:
                return f"choose {RNG.randrange(len(relics))}"
            return "proceed" if "proceed" in avail else "skip"
        if "proceed" in avail:
            return "proceed"
        if "confirm" in avail:
            return "confirm"
        return "wait 30"

    def main(self):
        print("ready", flush=True)
        while True:
            msg = recv()
            if "error" in msg:
                log("error", msg["error"])
                send("state")
                continue
            if not msg.get("in_game"):
                if self.run >= RUNS:
                    log("done")
                    while True:
                        recv()
                seed = "".join(RNG.choice(SEED_CHARS) for _ in range(10))
                self.out = open(OUT / f"run-{self.run:04d}.jsonl", "w")
                # The harness profile has the final act unlocked (since 2026-10-14).
                self.write({"type": "setup", "seed": seed, "final_act": True, "character": CHARACTER})
                self.sped_up = False
                self.buffed = False
                send(f"start {CHARACTER} 0 {seed}")
                continue
            if self.out is None:
                # A run we already finished (abandoning, or the death screen): keep clicking through.
                avail = msg.get("available_commands", [])
                send("proceed" if "proceed" in avail else "wait 30")
                continue
            if not self.sped_up:
                self.sped_up = True
                send('harness {"speed": %s, "ticks": %d}' % (SPEED, TICKS))
                continue
            self.write({"type": "state", "state": msg["game_state"], "available": msg.get("available_commands", [])})
            cmd = self.decide(msg)
            if cmd is None:
                continue
            self.write({"type": "command", "command": cmd})
            send(cmd)


if __name__ == "__main__":
    try:
        Recorder().main()
    except Exception as e:  # keep the error in the log; stdout belongs to the game
        log("crash", repr(e))
        raise
