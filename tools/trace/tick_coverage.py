#!/usr/bin/env python3
"""Ticks docs/checklist's real-game boxes from `sts-cli coverage` and lists what is left.

    ./target/release/sts-cli coverage harness/traces harness/soak/*/ > coverage.json
    python3 tools/trace/tick_coverage.py coverage.json          # report only
    python3 tools/trace/tick_coverage.py coverage.json --tick   # tick the boxes too

An item is ticked when the recordings and soak runs that match the game exercised it (see
crates/sts-cli/src/coverage.rs): a card played, a potion drunk, a power present, a relic whose
effect changed the outcome (or, for the four that do nothing, held through a matching fight or
run), a curse or status held in hand, a monster that took a turn, an event with an option taken. Existing
ticks are kept. The report lists every ported item still unticked.
"""
import json
import os
import re
import sys

CHECKLIST = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "checklist")
KINDS = {"cards": "cards", "relics": "relics", "potions": "potions", "powers": "powers", "monsters": "monsters", "events": "events"}
ROW = re.compile(r"^\| (?P<id>[^|]+?) \| (?P<detail>[^|]*?) \| (?P<status>[^|]*?) \| (?P<test>\[[ x]\]) \| (?P<game>\[[ x]\]) \|$")


def main():
    cov = json.load(open(sys.argv[1]))
    tick = "--tick" in sys.argv
    print("sources:", json.dumps(cov["sources"]))
    left_total = 0
    for kind, key in KINDS.items():
        path = os.path.join(CHECKLIST, kind + ".md")
        seen = set(cov[key])
        if kind == "cards":
            # Curses and statuses count when held (they can't be played).
            seen |= set(cov.get("cards_held", {}))
        out, ticked, left = [], 0, []
        section = ""
        for line in open(path).read().split("\n"):
            if line.startswith("## "):
                section = line[3:]
            m = ROW.match(line)
            if m and m.group("status") == "ported":
                if m.group("game") == "[ ]":
                    if m.group("id") in seen:
                        if tick:
                            line = line[: line.rfind("[ ]")] + "[x] |"
                        ticked += 1
                    else:
                        left.append(f"{section.split(' (')[0]}: {m.group('id')}")
            out.append(line)
        if tick:
            open(path, "w").write("\n".join(out))
        left_total += len(left)
        print(f"\n{kind}: {ticked} {'ticked' if tick else 'to tick'}, {len(left)} not exercised")
        for x in left:
            print("   ", x)
    print(f"\n{left_total} ported items not exercised by a matching recording")


if __name__ == "__main__":
    main()
