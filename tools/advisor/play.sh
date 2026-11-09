#!/bin/sh
# Play the real game with the network's advice in the top-left corner (F9 hides it).
#
#     tools/advisor/play.sh [checkpoint]
#
# Copies the checkpoint (if given) to harness/advisor/model.pt, points CommunicationMod at
# tools/advisor/advisor.py (starting it with the game), launches the harness game, and puts
# CommunicationMod's previous command back when the game closes. Needs the harness built
# (tools/trace/setup.sh) and the sts module (python/build.sh).
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CONF="$HOME/Library/Preferences/ModTheSpire/CommunicationMod/config.properties"
mkdir -p "$REPO/harness/advisor"
if [ -n "$1" ]; then
    cp "$1" "$REPO/harness/advisor/model.pt"
fi
[ -f "$REPO/harness/advisor/model.pt" ] || { echo "no checkpoint: pass one (e.g. a run's latest.pt)"; exit 1; }
[ -f "$CONF" ] || { echo "no CommunicationMod config at $CONF (run the game once with the mod)"; exit 1; }

cp "$CONF" "$CONF.before-advisor"
restore() { mv "$CONF.before-advisor" "$CONF"; echo "restored CommunicationMod config"; }
trap restore EXIT
python3 - "$CONF" "command=$REPO/.venv/bin/python $REPO/tools/advisor/advisor.py" <<'PY'
import sys
path, line = sys.argv[1], sys.argv[2]
lines = [l for l in open(path).read().splitlines() if not l.startswith(("command=", "runAtGameStart="))]
open(path, "w").write("\n".join(lines + [line, "runAtGameStart=true"]) + "\n")
PY
cd "$REPO/harness/game" && jre/bin/java -Xmx2G -jar ModTheSpire.jar --skip-launcher --mods basemod,CommunicationMod
