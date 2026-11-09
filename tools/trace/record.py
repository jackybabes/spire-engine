"""Record combat traces from the real game through CommunicationMod.

CommunicationMod launches this script and talks to it over stdin/stdout: it sends one JSON
game state per line whenever the game is waiting for input, and reads one command per line.
Stdout is the command channel, so everything else goes to the log file.

Each fight is set up with the local `harness` command (see harness/src/CommunicationMod
Harness.java): set the master deck and HP, then start the encounter. The script then plays
random legal moves until the fight ends, logging every state and command to
<out>/trace-<n>.jsonl.

Usage (as CommunicationMod's command):
    python3 -I record.py <out dir> <fights> <policy seed> [speed] [act2|keys] [hp] [--character silent|defect|watcher] [--cards A,B]
        [--relics A,B] [--potions A,B]
"""
import json
import os
import random
import sys
import time
from pathlib import Path

# --character ironclad|silent (anywhere on the line): who plays; the Ironclad by default.
CHARACTER = "ironclad"
if "--character" in sys.argv:
    i = sys.argv.index("--character")
    CHARACTER = sys.argv[i + 1].lower()
    del sys.argv[i:i + 2]
# --cards A,B_C (anywhere on the line): every deck gets a few of these cards (spaces as `_`, since
# CommunicationMod splits its command on whitespace), for cards earlier recordings missed.
FOCUS_CARDS = []
if "--cards" in sys.argv:
    i = sys.argv.index("--cards")
    FOCUS_CARDS = [c.replace("_", " ") for c in sys.argv[i + 1].split(",")]
    del sys.argv[i:i + 2]
# --relics A,B and --potions A,B (anywhere on the line): every fight also gets these (IDs, spaces
# as `_`), for items earlier batches never exercised.
FOCUS_RELICS, FOCUS_POTIONS = [], []
for flag, dest in (("--relics", FOCUS_RELICS), ("--potions", FOCUS_POTIONS)):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        dest.extend(x.replace("_", " ") for x in sys.argv[i + 1].split(","))
        del sys.argv[i:i + 2]
OUT = Path(sys.argv[1])
FIGHTS = int(sys.argv[2])
RNG = random.Random(int(sys.argv[3]))
SPEED = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
# Game updates per drawn frame (HARNESS_TICKS, the harness's "ticks"): above 1, every update is a
# fixed 1/60 s times the speed and a frame runs that many before drawing once, so the game goes
# through the same states as at normal speed, faster (HarnessTicksPatch). Use speed 1 with it.
TICKS = int(os.environ.get("HARNESS_TICKS", 1))
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "recorder.log", "a")

ENCOUNTERS = [
    "Cultist", "Jaw Worm", "2 Louse", "Small Slimes", "Blue Slaver", "Gremlin Gang", "Looter",
    "Large Slime", "Lots of Slimes", "Exordium Thugs", "Exordium Wildlife", "Red Slaver",
    "3 Louse", "2 Fungi Beasts", "Gremlin Nob", "Lagavulin", "3 Sentries", "The Guardian",
    "Hexaghost", "Slime Boss",
]
ACT2 = [
    "Spheric Guardian", "Chosen", "Shell Parasite", "3 Byrds", "2 Thieves", "Chosen and Byrds",
    "Sentry and Sphere", "Snake Plant", "Snecko", "Centurion and Healer", "Cultist and Chosen",
    "3 Cultists", "Shelled Parasite and Fungi", "Gremlin Leader", "Slavers", "Book of Stabbing",
    "Masked Bandits", "Colosseum Nobs", "Colosseum Slavers", "Automaton", "Collector", "Champ",
]
ACT3 = [
    "3 Darklings", "Orb Walker", "3 Shapes", "Spire Growth", "Transient", "4 Shapes", "Maw",
    "Sphere and 2 Shapes", "Jaw Worm Horde", "Writhing Mass", "Giant Head", "Nemesis", "Reptomancer",
    "2 Orb Walkers", "Mysterious Sphere", "Awakened One", "Time Eater", "Donu and Deca",
]
# Optional 5th argument: "act2", "act3", or a comma-separated list of encounter keys (spaces as
# underscores, since CommunicationMod splits its command on whitespace); 6th: player HP.
HP = int(sys.argv[6]) if len(sys.argv) > 6 else {"ironclad": 80, "silent": 70, "defect": 75, "watcher": 72}[CHARACTER]
if len(sys.argv) > 5:
    ENCOUNTERS = {"act2": ACT2, "act3": ACT3}.get(sys.argv[5]) or [e.replace("_", " ") for e in sys.argv[5].split(",")]
STARTER = [["Strike_R", 0]] * 5 + [["Defend_R", 0]] * 4 + [["Bash", 0]]
# Every non-basic Ironclad card except Infernal Blade (its pool depends on unlocks).
RED = """Anger Armaments Barricade Battle_Trance Berserk Blood_for_Blood Bloodletting Bludgeon
Body_Slam Brutality Burning_Pact Carnage Clash Cleave Clothesline Combust Corruption Dark_Embrace
Demon_Form Disarm Double_Tap Dropkick Dual_Wield Entrench Evolve Exhume Feed Feel_No_Pain Fiend_Fire
Fire_Breathing Flame_Barrier Flex Ghostly_Armor Havoc Headbutt Heavy_Blade Hemokinesis Immolate
Impervious Inflame Intimidate Iron_Wave Juggernaut Limit_Break Metallicize Offering Perfected_Strike
Pommel_Strike Power_Through Pummel Rage Rampage Reaper Reckless_Charge Rupture Searing_Blow
Second_Wind Seeing_Red Sentinel Sever_Soul Shockwave Shrug_It_Off Spot_Weakness Sword_Boomerang
Thunderclap True_Grit Twin_Strike Uppercut Warcry Whirlwind Wild_Strike""".split()
RED = [c.replace("_", " ") for c in RED]
# Every non-basic Silent card (card IDs, which the harness command takes).
GREEN = ['A Thousand Cuts', 'Accuracy', 'Acrobatics', 'Adrenaline', 'After Image', 'Venomology', 'All Out Attack', 'Backflip', 'Backstab', 'Bane', 'Blade Dance', 'Blur', 'Bouncing Flask', 'Bullet Time', 'Burst', 'Calculated Gamble', 'Caltrops', 'Catalyst', 'Choke', 'Cloak And Dagger', 'Concentrate', 'Corpse Explosion', 'Crippling Poison', 'Dagger Spray', 'Dagger Throw', 'Dash', 'Deadly Poison', 'Deflect', 'Die Die Die', 'Distraction', 'Dodge and Roll', 'Doppelganger', 'Endless Agony', 'Envenom', 'Escape Plan', 'Eviscerate', 'Expertise', 'Finisher', 'Flechettes', 'Flying Knee', 'Footwork', 'Glass Knife', 'Grand Finale', 'Heel Hook', 'Infinite Blades', 'Leg Sweep', 'Malaise', 'Masterful Stab', 'Night Terror', 'Noxious Fumes', 'Outmaneuver', 'Phantasmal Killer', 'PiercingWail', 'Poisoned Stab', 'Predator', 'Prepared', 'Quick Slash', 'Reflex', 'Riddle With Holes', 'Setup', 'Skewer', 'Slice', 'Underhanded Strike', 'Storm of Steel', 'Sucker Punch', 'Tactician', 'Terror', 'Tools of the Trade', 'Unload', 'Well Laid Plans', 'Wraith Form v2']
SILENT_STARTER = [["Strike_G", 0]] * 5 + [["Defend_G", 0]] * 5 + [["Survivor", 0], ["Neutralize", 0]]
# Every non-basic Defect card.
BLUE = ['Aggregate', 'All For One', 'Amplify', 'Auto Shields', 'Ball Lightning', 'Barrage', 'Beam Cell', 'Biased Cognition', 'Blizzard', 'BootSequence', 'Buffer', 'Capacitor', 'Chaos', 'Chill', 'Gash', 'Cold Snap', 'Compile Driver', 'Conserve Battery', 'Consume', 'Coolheaded', 'Core Surge', 'Creative AI', 'Darkness', 'Defragment', 'Doom and Gloom', 'Double Energy', 'Echo Form', 'Electrodynamics', 'Undo', 'FTL', 'Fission', 'Force Field', 'Fusion', 'Genetic Algorithm', 'Glacier', 'Go for the Eyes', 'Heatsinks', 'Hello World', 'Hologram', 'Hyperbeam', 'Leap', 'Lockon', 'Loop', 'Machine Learning', 'Melter', 'Meteor Strike', 'Multi-Cast', 'Steam Power', 'Rainbow', 'Reboot', 'Rebound', 'Redo', 'Recycle', 'Reinforced Body', 'Reprogram', 'Rip and Tear', 'Scrape', 'Seek', 'Self Repair', 'Skim', 'Stack', 'Static Discharge', 'Steam', 'Storm', 'Streamline', 'Sunder', 'Sweeping Beam', 'Tempest', 'Thunder Strike', 'Turbo', 'White Noise']
DEFECT_STARTER = [["Strike_B", 0]] * 4 + [["Defend_B", 0]] * 4 + [["Zap", 0], ["Dualcast", 0]]
# Every non-basic Watcher card.
PURPLE = ['Adaptation', 'Alpha', 'BattleHymn', 'Blasphemy', 'BowlingBash', 'Brilliance', 'CarveReality', 'ClearTheMind', 'Collect', 'Conclude', 'ConjureBlade', 'Consecrate', 'Crescendo', 'CrushJoints', 'CutThroughFate', 'DeceiveReality', 'DeusExMachina', 'DevaForm', 'Devotion', 'EmptyBody', 'EmptyFist', 'EmptyMind', 'Establishment', 'Evaluate', 'Fasting2', 'FearNoEvil', 'FlurryOfBlows', 'FlyingSleeves', 'FollowUp', 'ForeignInfluence', 'Halt', 'Indignation', 'InnerPeace', 'Judgement', 'JustLucky', 'LessonLearned', 'LikeWater', 'MasterReality', 'Meditate', 'MentalFortress', 'Nirvana', 'Omniscience', 'PathToVictory', 'Perseverance', 'Pray', 'Prostrate', 'Protect', 'Ragnarok', 'ReachHeaven', 'Sanctity', 'SandsOfTime', 'SashWhip', 'Scrawl', 'SignatureMove', 'SpiritShield', 'Study', 'Swivel', 'TalkToTheHand', 'Tantrum', 'ThirdEye', 'Vault', 'Vengeance', 'Wallop', 'WaveOfTheHand', 'Weave', 'WheelKick', 'WindmillStrike', 'Wireheading', 'Wish', 'Worship', 'WreathOfFlame']
WATCHER_STARTER = [["Strike_P", 0]] * 4 + [["Defend_P", 0]] * 4 + [["Eruption", 0], ["Vigilance", 0]]
if CHARACTER == "silent":
    STARTER, RED = SILENT_STARTER, GREEN
if CHARACTER == "defect":
    STARTER, RED = DEFECT_STARTER, BLUE
if CHARACTER == "watcher":
    STARTER, RED = WATCHER_STARTER, PURPLE
# Colorless cards, curses and statuses --cards may also name (any character).
OTHER_CARDS = ['Apotheosis', 'Bandage Up', 'Blind', 'Chrysalis', 'Dark Shackles', 'Deep Breath', 'Discovery', 'Dramatic Entrance',
               'Enlightenment', 'Finesse', 'Flash of Steel', 'Forethought', 'Ghostly', 'Good Instincts', 'HandOfGreed', 'Impatience',
               'J.A.X.', 'Jack Of All Trades', 'Madness', 'Magnetism', 'Master of Strategy', 'Mayhem', 'Metamorphosis', 'Mind Blast',
               'Panacea', 'Panache', 'PanicButton', 'Purity', 'RitualDagger', 'Sadistic Nature', 'Secret Technique', 'Secret Weapon',
               'Swift Strike', 'The Bomb', 'Thinking Ahead', 'Transmutation', 'Trip', 'Violence',
               'AscendersBane', 'Clumsy', 'CurseOfTheBell', 'Decay', 'Doubt', 'Injury', 'Necronomicurse', 'Normality', 'Pain',
               'Parasite', 'Pride', 'Regret', 'Shame', 'Writhe', 'Burn', 'Dazed', 'Slimed', 'Void', 'Wound']
# The harness only knows library cards (Impulse, say, is in none and stalls the session).
ANY_COLOR = set(RED) | set(GREEN) | set(BLUE) | set(PURPLE) | set(OTHER_CARDS)
for c in FOCUS_CARDS:
    if c not in ANY_COLOR:
        sys.exit(f"--cards: {c} is not a library card")


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


# Relics with combat effects (no selection screens), and the Ironclad potions (discovery ones
# included: HarnessDiscoveryPatch makes the game roll their choices once, like the engine).
RELICS = ['Akabeko', 'Anchor', 'Ancient Tea Set', 'Art of War', 'Bag of Marbles', 'Bag of Preparation', 'Blood Vial', 'Boot', 'Bronze Scales', 'Centennial Puzzle', 'Happy Flower', 'Lantern', 'Nunchaku', 'Oddly Smooth Stone', 'Orichalcum', 'Pen Nib', 'PreservedInsect', 'Vajra', 'Red Skull', 'Blue Candle', 'Gremlin Horn', 'HornCleat', 'InkBottle', 'Kunai', 'Letter Opener', 'Meat on the Bone', 'Mercury Hourglass', 'Mummified Hand', 'Ornamental Fan', 'Pantograph', 'Shuriken', 'StrikeDummy', 'Self Forming Clay', 'Paper Frog', 'Bird Faced Urn', 'Calipers', 'CaptainsWheel', 'Dead Branch', 'FossilizedHelix', 'Gambling Chip', 'Ginger', 'Ice Cream', 'Incense Burner', 'Lizard Tail', 'Pocketwatch', 'StoneCalendar', 'Thread and Needle', 'Torii', 'TungstenRod', 'Turnip', 'Unceasing Top', 'Champion Belt', "Charon's Ashes", 'Magic Flower', 'TheAbacus', 'ClockworkSouvenir', 'HandDrill', 'Medical Kit', 'OrangePellets', 'Sling', 'Strange Spoon', 'Toolbox', 'Brimstone', 'Chemical X', 'Sundial', 'Runic Cube', 'Runic Dome', 'Runic Pyramid', 'Snecko Eye', 'Busted Crown', 'Coffee Dripper', 'Cursed Key', 'Ectoplasm', 'Fusion Hammer', 'Mark of Pain', "Philosopher's Stone", 'Velvet Choker', 'SlaversCollar', 'Black Blood', 'GremlinMask', 'CultistMask', 'Red Mask', 'MutagenicStrength', 'NeowsBlessing', 'Odd Mushroom', 'Mark of the Bloom']
SILENT_RELICS = ['Snake Skull', 'Paper Crane', 'Ninja Scroll', 'The Specimen', 'Tingsha', 'Tough Bandages', 'TwistedFunnel', 'WristBlade', 'HoveringKite']
# Not Frozen Core: it replaces Cracked Core (it's the starter now and then, below), as Ring of the
# Serpent replaces Ring of the Snake.
DEFECT_RELICS = ['Cables', 'DataDisk', 'Emotion Chip', 'Inserter', 'Nuclear Battery', 'Runic Capacitor', 'Symbiotic Virus']
# Not Holy Water: it replaces Pure Water (the starter now and then, below).
WATCHER_RELICS = ['CloakClasp', 'Damaru', 'GoldenEye', 'Melange', 'TeardropLocket', 'VioletLotus', 'Yang']
POTIONS = ['AttackPotion', 'SkillPotion', 'PowerPotion', 'ColorlessPotion', 'BloodPotion', 'ElixirPotion', 'HeartOfIron', 'Block Potion', 'Dexterity Potion', 'Energy Potion', 'Explosive Potion', 'Fire Potion', 'Strength Potion', 'Swift Potion', 'Weak Potion', 'FearPotion', 'SteroidPotion', 'SpeedPotion', 'BlessingOfTheForge', 'Regen Potion', 'Ancient Potion', 'LiquidBronze', 'GamblersBrew', 'EssenceOfSteel', 'DuplicationPotion', 'DistilledChaos', 'LiquidMemories', 'CultistPotion', 'Fruit Juice', 'SneckoOil', 'FairyPotion', 'SmokeBomb', 'EntropicBrew']


CLASS_ITEMS = {
    "silent": (SILENT_RELICS, ["Poison Potion", "CunningPotion", "GhostInAJar"]),
    "defect": (DEFECT_RELICS, ["FocusPotion", "PotionOfCapacity", "EssenceOfDarkness"]),
    "watcher": (WATCHER_RELICS, ["Ambrosia", "BottledMiracle", "StancePotion"]),
}
if CHARACTER in CLASS_ITEMS:
    RELICS = [r for r in RELICS if r != "Black Blood"] + CLASS_ITEMS[CHARACTER][0]
    POTIONS = [p for p in POTIONS if p not in ("BloodPotion", "ElixirPotion", "HeartOfIron")] + CLASS_ITEMS[CHARACTER][1]
STARTER_RELIC = {"ironclad": "Burning Blood", "silent": "Ring of the Snake", "defect": "Cracked Core", "watcher": "PureWater"}[CHARACTER]


CLASS_RELICS = CLASS_ITEMS.get(CHARACTER, ([], []))[0]
CLASS_POTIONS = CLASS_ITEMS.get(CHARACTER, ([], []))[1]


def random_relics():
    picked = pick_relics()
    return picked + [r for r in FOCUS_RELICS if r not in picked]


def pick_relics():
    if not CLASS_RELICS:
        return [STARTER_RELIC] + RNG.sample(RELICS, RNG.randint(0, 4))
    # Other characters: mostly their own relics, so the recordings exercise them.
    upgrade = {"silent": "Ring of the Serpent", "defect": "FrozenCore", "watcher": "HolyWater"}.get(CHARACTER)
    starter = upgrade if upgrade and RNG.random() < 0.25 else STARTER_RELIC
    shared = [r for r in RELICS if r not in CLASS_RELICS]
    return [starter] + RNG.sample(CLASS_RELICS, RNG.randint(1, 3)) + RNG.sample(shared, RNG.randint(0, 2))


def random_potions():
    return (FOCUS_POTIONS + pick_potions())[:3]


def pick_potions():
    if CLASS_POTIONS:
        return [RNG.choice(CLASS_POTIONS) if RNG.random() < 0.6 else RNG.choice(POTIONS) for _ in range(RNG.randint(1, 3))]
    return [RNG.choice(POTIONS) for _ in range(RNG.randint(0, 3))]


def random_deck():
    if RNG.random() < 0.3 and not FOCUS_CARDS:
        return STARTER
    deck = [list(c) for c in STARTER[:RNG.randint(3, 10)]]
    for c in RNG.sample(FOCUS_CARDS, min(len(FOCUS_CARDS), RNG.randint(2, 4))):
        deck.append([c, 1 if RNG.random() < 0.4 else 0])
    for _ in range(RNG.randint(3, 12)):
        deck.append([RNG.choice(RED), 1 if RNG.random() < 0.4 else 0])
    RNG.shuffle(deck)
    return deck


def combat_options(gs):
    """Commands available in a combat decision, mirroring what the engine considers legal."""
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


def main():
    print("ready", flush=True)
    fight = 0
    out = None
    in_fight = False
    smoked = False
    while True:
        msg = recv()
        if "error" in msg:
            log("error", msg["error"])
        avail = msg.get("available_commands", [])
        if not msg.get("in_game"):
            if fight >= FIGHTS:
                log("done")
                while True:
                    recv()
            send(f"start {CHARACTER} 0")
            continue
        gs = msg["game_state"]
        screen = gs.get("screen_type")
        in_combat = gs.get("room_phase") == "COMBAT" and "combat_state" in gs
        if not in_combat:
            # See record_run.py: after a Smoke Bomb, wait out the escape.
            smoked = False
        if in_combat and out is not None:
            out.write(json.dumps({"type": "state", "state": gs, "available": avail}) + "\n")
        if not in_combat and in_fight:
            # The fight ended: record the final state and close the trace.
            out.write(json.dumps({"type": "end", "state": gs}) + "\n")
            out.close()
            out = None
            in_fight = False
            fight += 1
            log("fight finished", fight)
        if screen == "GAME_OVER":
            if in_fight and out is not None:
                out.write(json.dumps({"type": "end", "state": gs}) + "\n")
                out.close()
                out = None
                in_fight = False
                fight += 1
                log("fight finished (death)", fight)
            send("proceed")
            continue
        if not in_combat:
            if fight >= FIGHTS:
                log("done")
                while True:
                    recv()
            enc = ENCOUNTERS[fight % len(ENCOUNTERS)]
            deck = random_deck()
            setup = {"deck": deck, "hp": HP, "max_hp": HP, "encounter": enc, "speed": SPEED, "ticks": TICKS,
                     "relics": random_relics(), "potions": random_potions()}
            out = open(OUT / f"trace-{fight:04d}.jsonl", "w")
            out.write(json.dumps({"type": "setup", "setup": setup}) + "\n")
            in_fight = True
            send("harness " + json.dumps(setup))
            continue
        # In combat, waiting for a decision.
        ss = gs.get("screen_state", {})
        usable = [(i, p) for i, p in enumerate(gs.get("potions", [])) if p.get("can_use")]
        if smoked:
            cmd = "wait 30"
        elif screen in ("HAND_SELECT", "GRID") and "choose" not in avail and "confirm" in avail:
            cmd = "confirm"
        elif screen == "HAND_SELECT" and "confirm" in avail and (ss.get("selected") or (ss.get("can_pick_zero") and RNG.random() < 0.5)):
            cmd = "confirm"
        elif "potion" in avail and "end" in avail and usable and RNG.random() < 0.15:
            i, p = RNG.choice(usable)
            smoked = p["id"] == "SmokeBomb"
            if p.get("requires_target"):
                targets = [t for t, m in enumerate(gs["combat_state"]["monsters"]) if not m["is_gone"] and not m.get("half_dead")]
                cmd = f"potion use {i} {RNG.choice(targets)}"
            else:
                cmd = f"potion use {i}"
        elif "choose" in avail and screen == "CARD_REWARD":
            cmd = f"choose {RNG.randrange(len(gs['choice_list']))}"
        elif "choose" in avail and screen == "GRID":
            # Picking a selected card again would unselect it: pick among the others. A grid of
            # any number (Scry) stops at a random point.
            picked = {c.get("uuid") for c in ss.get("selected_cards", [])}
            free = [i for i, c in enumerate(ss.get("cards", [])) if c.get("uuid") not in picked]
            done = len(picked) >= ss.get("num_cards", 1)
            if "confirm" in avail and (not free or done or (ss.get("any_number") and RNG.random() < 0.35)):
                cmd = "confirm"
            elif free:
                cmd = f"choose {RNG.choice(free)}"
            else:
                cmd = "wait 30"
        elif "choose" in avail and screen == "HAND_SELECT":
            cmd = f"choose {RNG.randrange(len(gs['choice_list']))}"
        elif "play" in avail or "end" in avail:
            cmd = RNG.choice(combat_options(gs))
        elif "confirm" in avail:
            cmd = "confirm"
        else:
            cmd = "wait 30"
        out.write(json.dumps({"type": "command", "command": cmd}) + "\n")
        send(cmd)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # keep the error in the log; stdout belongs to the game
        log("crash", repr(e))
        raise
