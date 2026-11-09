"""The port checklist (docs/checklist/): every card, relic, potion, power, monster, event, orb,
stance, character, blight and daily mod in the decompiled game, with the engine's status and a
unit-test and a real-game box for each.

    python3 tools/extract/port_checklist.py [reference/src/com/megacrit/cardcrawl] [docs/checklist]

Engine status, read from crates/sts-core:
- ported: the engine has it and its logic (Ironclad's cards and relics, every shared relic and
  potion, every power, monster and event the engine knows);
- data only: in the engine's generated data tables, but its logic isn't written yet (other
  characters' cards and relics not yet in `cards::PORTED` and `relics::PORTED`, and potions);
- missing: not in the engine at all.
Run it again after porting something: the ticked boxes in the existing files are kept (matched by
ID and class), the statuses are worked out afresh. It writes IDs and types only, never game source.
"""
import collections
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "reference/src/com/megacrit/cardcrawl")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "docs/checklist")
ENGINE = os.path.join(ROOT, "crates/sts-core/src")


def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def java_files(*dirs):
    for d in dirs:
        p = os.path.join(REF, d)
        if not os.path.isdir(p):
            continue
        for name in sorted(os.listdir(p)):
            if name.endswith(".java") and not name.startswith("Abstract"):
                yield d, name[:-5], read(os.path.join(p, name))


def first(pattern, text, default=""):
    m = re.search(pattern, text)
    return m.group(1) if m else default


def engine_strings(*rel_paths, pattern):
    out = set()
    for rel in rel_paths:
        p = os.path.join(ENGINE, rel)
        paths = [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.endswith(".rs")] if os.path.isdir(p) else [p]
        for path in paths:
            out.update(re.findall(pattern, read(path)))
    return out


def engine_table(rel, kind):
    """{java_id: class or color} from a generated data table."""
    out = {}
    for line in read(os.path.join(ENGINE, rel)).splitlines():
        jid = first(r'java_id: "([^"]+)"', line)
        if jid:
            out[jid] = first(kind + r"::(\w+)", line)
    return out


# ---- the engine ----
E_CARDS = engine_table("cards/data.rs", "CardColor")
CARD_VARIANTS = dict(re.findall(r'id: CardId::(\w+), java_id: "([^"]+)"', read(os.path.join(ENGINE, "cards/data.rs"))))
CARD_VARIANTS = {jid: v for v, jid in CARD_VARIANTS.items()}
_ported = re.search(r"pub const PORTED: &\[CardId\] = &\[(.*?)\];", read(os.path.join(ENGINE, "cards/mod.rs")), re.S)
PORTED_CARDS = set(re.findall(r"CardId::(\w+)", _ported.group(1))) if _ported else set()
_ported = re.search(r"pub const PORTED: &\[RelicId\] = &\[(.*?)\];", read(os.path.join(ENGINE, "relics/mod.rs")), re.S)
PORTED_RELICS = set(re.findall(r"RelicId::(\w+)", _ported.group(1))) if _ported else set()
RELIC_VARIANTS = {jid: v for v, jid in re.findall(r'id: RelicId::(\w+), java_id: "([^"]+)"', read(os.path.join(ENGINE, "relics/data.rs")))}
# The Ironclad's groups: everything in the data table has its logic (CardId::is_ported).
FULL_CARD_GROUPS = {"red", "colorless", "curses", "status"}
UNUSED_CARDS = {"Impulse": "in no card library and made by nothing", "Discipline": "in no card library and made by nothing",
                "Unraveling": "in no card library and made by nothing"}
E_RELICS = engine_table("relics/data.rs", "RelicClass")
E_POTIONS = engine_table("potions/data.rs", "PotionRarity")
POTION_VARIANTS = {first(r'java_id: "([^"]+)"', l): first(r"PotionId::(\w+)", l)
                   for l in read(os.path.join(ENGINE, "potions/data.rs")).splitlines() if "java_id" in l}
POTION_LOGIC = read(os.path.join(ENGINE, "combat/potions.rs")) + read(os.path.join(ENGINE, "run/mod.rs"))
E_POWERS = engine_strings("powers.rs", pattern=r'=> "([^"]+)"')
E_MONSTERS = engine_strings("monsters/mod.rs", pattern=r'=> "([^"]+)"')
E_EVENTS = engine_strings("run/events.rs", "run/events_city.rs", "run/events_beyond.rs", pattern=r'=> "([^"]+)"')
# Known gaps among things the engine has: listed so the checklist doesn't call them done.
GAPS = {}
# Classes in the game's code that no game reaches (status "n/a").
NOT_IN_GAME = {
    "ApologySlime": "MonsterHelper's fallback for an unknown encounter name; never met",
    "HexaghostBody": "Hexaghost's drawn body, not a monster",
    "HexaghostOrb": "Hexaghost's drawn orbs, not a monster",
    "SpireHeart": "the victory room after Act 3 without the keys; no decision (the engine ends the run)",
}
LIBRARY = read(os.path.join(REF, "helpers/RelicLibrary.java"))

# Which character's potions (PotionHelper.getPotions).
POTION_CLASS = {}
helper = read(os.path.join(REF, "helpers/PotionHelper.java"))
switch = helper[helper.index("switch (c)"):helper.index("} else {")]
NAMES = {"IRONCLAD": "Ironclad", "THE_SILENT": "Silent", "DEFECT": "Defect", "WATCHER": "Watcher"}
for cls, body in re.findall(r"case (\w+):(.*?)(?=case |\Z)", switch, re.S):
    for pid in re.findall(r'retVal\.add\("([^"]+)"\)', body):
        POTION_CLASS[pid] = NAMES[cls]
CHARACTER = {"Red": "Ironclad", "Green": "Silent", "Blue": "Defect", "Purple": "Watcher", "Shared": "shared"}


# ---- the game ----
def cards():
    rows = []
    for d, cls, src in java_files("cards/red", "cards/green", "cards/blue", "cards/purple", "cards/colorless",
                                  "cards/curses", "cards/status", "cards/tempCards", "cards/optionCards"):
        cid = first(r'public static final String ID = "([^"]+)"', src) or first(r'super\(\s*"([^"]+)"', src)
        if not cid:
            continue
        ctype = first(r"CardType\.(\w+)", src).title()
        rarity = first(r"CardRarity\.(\w+)", src).title()
        group = d.split("/")[1]
        if cid in UNUSED_CARDS:
            status = "n/a: " + UNUSED_CARDS[cid]
        elif cid not in E_CARDS:
            status = "missing"
        elif group in FULL_CARD_GROUPS or CARD_VARIANTS[cid] in PORTED_CARDS:
            status = "ported"
        else:
            status = "data only"
        rows.append((group, cid, f"{ctype}, {rarity}", status))
    return rows


def relics():
    rows = []
    for _, cls, src in java_files("relics"):
        rid = first(r'public static final String ID = "([^"]+)"', src)
        if not rid:
            continue
        tier = first(r"RelicTier\.(\w+)", src).title()
        klass = E_RELICS.get(rid)
        who = CHARACTER.get(klass, "?") if klass else "?"
        if klass is None:
            status = "missing"
        elif klass in ("Shared", "Red") or RELIC_VARIANTS.get(rid) in PORTED_RELICS:
            status = "ported"
        else:
            status = "data only"
        note = GAPS.get(cls, "")
        if note:
            status = "gap: " + note
        if not re.search(r"new " + cls + r"\(", LIBRARY):
            status = "n/a: not in the game's relic library"
            who = "unused"
        rows.append((who, rid, tier, status))
    return rows


def potions():
    rows = []
    for _, cls, src in java_files("potions"):
        pid = first(r'public static final String POTION_ID = "([^"]+)"', src)
        if not pid:
            continue
        rarity = first(r"PotionRarity\.(\w+)", src).title()
        if rarity == "Placeholder":
            continue  # PotionSlot: an empty slot, not a potion
        who = POTION_CLASS.get(pid, "shared")
        if pid not in E_POTIONS:
            status = "missing"
        else:
            variant = POTION_VARIANTS.get(pid, "")
            status = "ported" if who in ("shared", "Ironclad") or re.search(r"\b" + re.escape(variant) + r"\b", POTION_LOGIC) else "data only"
        rows.append((who, pid, rarity, status))
    return rows


def created_classes():
    """Classes some code in the game constructs (`new X(`), outside their own file."""
    out = collections.Counter()
    for root, _, names in os.walk(REF):
        for name in names:
            if name.endswith(".java"):
                own = name[:-5]
                for cls in set(re.findall(r"new (\w+)\(", read(os.path.join(root, name)))):
                    if cls != own:
                        out[cls] += 1
    return out


def powers():
    rows = []
    made = created_classes()
    for d, cls, src in java_files("powers", "powers/watcher", "powers/deprecated"):
        pid = first(r'public static final String POWER_ID = "([^"]+)"', src)
        if not pid:
            continue
        kind = "Watcher" if d.endswith("watcher") else "deprecated" if d.endswith("deprecated") else ""
        if kind == "deprecated":
            status = "n/a: deprecated, unused"
        elif not made[cls]:
            # Before the ID check: ResurrectPower shares RegrowPower's ID ("Life Link").
            status = "n/a: created by nothing"
        elif pid in E_POWERS:
            status = "ported"
        else:
            status = "missing"
        rows.append((kind or "general", pid, cls, status))
    return rows


def monsters():
    rows = []
    for d, cls, src in java_files("monsters/exordium", "monsters/city", "monsters/beyond", "monsters/ending"):
        mid = first(r'public static final String ID = "([^"]+)"', src)
        if mid:
            status = "n/a: " + NOT_IN_GAME[cls] if cls in NOT_IN_GAME else "ported" if mid in E_MONSTERS else "missing"
            rows.append((d.split("/")[1], mid, cls, status))
    return rows


def events():
    rows = []
    for d, cls, src in java_files("events/exordium", "events/city", "events/beyond", "events/shrines"):
        eid = first(r'public static final String ID = "([^"]+)"', src)
        if eid:
            status = "n/a: " + NOT_IN_GAME[cls] if cls in NOT_IN_GAME else "ported" if eid in E_EVENTS else "missing"
            rows.append((d.split("/")[1], eid, cls, status))
    return rows


def simple(directory, pattern, group, ported=()):
    rows = []
    for _, cls, src in java_files(directory):
        sid = first(pattern, src)
        if sid:
            rows.append((group, sid, cls, "ported" if sid in ported or cls in ported else "missing"))
    return rows


def other():
    rows = []
    rows += simple("characters", r"PlayerClass\.(\w+)", "character", ported={"Ironclad", "TheSilent", "Defect", "Watcher"})
    # The engine's OrbId java ids; an empty slot is `None` in the player's slots.
    orbs_rs = os.path.join(ENGINE, "combat/orbs.rs")
    ported_orbs = set(re.findall(r'OrbId::\w+ => "(\w+)"', read(orbs_rs))) | {"Empty"} if os.path.exists(orbs_rs) else set()
    rows += simple("orbs", r'public static final String ORB_ID = "([^"]+)"', "orb", ported=ported_orbs)
    # The engine's Stance java ids (combat/stance.rs).
    stance_rs = os.path.join(ENGINE, "combat/stance.rs")
    ported_stances = set(re.findall(r'Stance::\w+ => "(\w+)"', read(stance_rs))) if os.path.exists(stance_rs) else {"Neutral"}
    rows += simple("stances", r'public static final String STANCE_ID = "([^"]+)"', "stance", ported=ported_stances)
    rows += simple("blights", r'public static final String ID = "([^"]+)"', "blight (Endless, optional)")
    rows += simple("daily/mods", r'public static final String ID = "([^"]+)"', "daily mod (optional)")
    return rows


# ---- writing ----
BOX = re.compile(r"^\| (?P<id>[^|]+?) \| (?P<detail>[^|]*?) \|.*\| (?P<test>\[[ x]\]) \| (?P<game>\[[ x]\]) \|$")


def old_boxes(path):
    """Ticks by (ID, detail), so two rows sharing an ID (a deprecated power and its successor) keep
    their own, and by ID alone, for a row whose detail column changed."""
    if not os.path.exists(path):
        return {}
    out = {}
    for line in read(path).splitlines():
        m = BOX.match(line)
        if m:
            boxes = (m.group("test"), m.group("game"))
            out[(m.group("id"), m.group("detail"))] = boxes
            out.setdefault(m.group("id"), boxes)
    return out


TITLES = {"cards": "Cards", "relics": "Relics", "potions": "Potions", "powers": "Powers", "monsters": "Monsters",
          "events": "Events", "other": "Characters, orbs, stances, blights, daily mods"}
GROUP_HEAD = {"cards": "Color", "relics": "Character", "potions": "Character", "powers": "Group", "monsters": "Act",
              "events": "Act", "other": "Kind"}
DETAIL_HEAD = {"cards": "Type, rarity", "relics": "Tier", "potions": "Rarity", "powers": "Class", "monsters": "Class",
               "events": "Class", "other": "Class"}


def write(name, rows):
    path = os.path.join(OUT, name + ".md")
    keep = old_boxes(path)
    counts = collections.Counter(r[3].split(":")[0] for r in rows)
    lines = [f"# {TITLES[name]}", "",
             "Generated by `tools/extract/port_checklist.py` (statuses from the engine; the boxes are kept when it runs",
             "again). Tick **unit test** when a test in the engine checks the item's behaviour against the game's code,",
             "and **real game** when a recording, playback or soak has exercised it and matched. See `README.md`.", "",
             f"{len(rows)} in all: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) + ".", ""]
    groups = collections.OrderedDict()
    order = ["shared", "Ironclad", "Silent", "Defect", "Watcher", "unused"]
    for r in sorted(rows, key=lambda r: order.index(r[0]) if r[0] in order else -1):
        groups.setdefault(r[0], []).append(r)
    for group, rs in groups.items():
        done = sum(1 for r in rs if r[3] == "ported")
        lines += [f"## {group} ({done} of {len(rs)} ported)", "",
                  f"| ID | {DETAIL_HEAD[name]} | Engine | Unit test | Real game |", "|---|---|---|---|---|"]
        for _, rid, detail, status in sorted(rs, key=lambda r: r[1].lower()):
            test, game = keep.get((rid, detail)) or keep.get(rid, ("[ ]", "[ ]"))
            lines.append(f"| {rid} | {detail} | {status} | {test} | {game} |")
        lines.append("")
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return counts


def main():
    tables = {"cards": cards(), "relics": relics(), "potions": potions(), "powers": powers(),
              "monsters": monsters(), "events": events(), "other": other()}
    summary = []
    for name, rows in tables.items():
        counts = write(name, rows)
        summary.append((name, len(rows), counts))
    width = max(len(n) for n, _, _ in summary)
    for name, total, counts in summary:
        print(f"{name:{width}s} {total:4d}  " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))


if __name__ == "__main__":
    main()
