"""Print card IDs in the order CardLibrary.initialize() inserts them into its HashMap.

Usage: python3 -I card_add_order.py <reference/src/com/megacrit/cardcrawl>
"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
lib = (root / "helpers/CardLibrary.java").read_text()

# Map class simple name -> ID constant, scanning every card source file.
ids = {}
for p in (root / "cards").rglob("*.java"):
    m = re.search(r'public static final String ID = "([^"]+)";', p.read_text())
    if m:
        ids.setdefault(p.stem, m.group(1))

init = re.search(r"public static void initialize\(\) \{(.*?)\n   \}", lib, re.S).group(1)
for fn in re.findall(r"(add\w+Cards)\(\);", init):
    body = re.search(r"private static void " + fn + r"\(\) \{(.*?)\n   \}", lib, re.S).group(1)
    for cls in re.findall(r"add\(new (\w+)\(", body):
        print(ids[cls])
