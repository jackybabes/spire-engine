#!/bin/sh
# Watch the network play in the real game (on the Mac), in one command:
#
#     tools/trace/watch.sh [file.json] [delay seconds]
#
# The file is either saved fights (python/watch/watch_fights.py, played by playback_fights.py) or a
# saved run (python/watch/watch_runs.py or `sts-cli run --save`, played by playback.py). Default:
# python/results/watch/fights.json, 0.8 s between moves. Points CommunicationMod at the right
# playback script, launches the modded game, and puts CommunicationMod's previous command back
# when the game closes.
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
FIGHTS=${1:-$REPO/python/results/watch/fights.json}
FIGHTS=$(cd "$(dirname "$FIGHTS")" && pwd)/$(basename "$FIGHTS")
DELAY=${2:-0.8}
CONF="$HOME/Library/Preferences/ModTheSpire/CommunicationMod/config.properties"
PY=$(command -v /opt/homebrew/bin/python3 || command -v python3)

[ -f "$FIGHTS" ] || { echo "no fights file at $FIGHTS"; exit 1; }
[ -f "$CONF" ] || { echo "no CommunicationMod config at $CONF (run the game with CommunicationMod once)"; exit 1; }
[ -d "$REPO/harness/game" ] || { echo "no harness at $REPO/harness/game (tools/trace/setup.sh builds it)"; exit 1; }

cp "$CONF" "$CONF.before-watch"
trap 'mv "$CONF.before-watch" "$CONF"; echo "restored CommunicationMod config"' EXIT
if "$PY" -c 'import json, sys; sys.exit(0 if "fights" in json.load(open(sys.argv[1])) else 1)' "$FIGHTS"; then
    SCRIPT=playback_fights.py
else
    SCRIPT=playback.py
    # A saved run is played with the game commands the current engine gives for its decisions
    # (sts-cli commands), not the ones saved with it, so engine fixes to them apply.
    (cd "$REPO" && cargo build --release -q -p sts-cli)
    mkdir -p "$REPO/harness/watch"
    OUT="$REPO/harness/watch/$(basename "${FIGHTS%.json}").cmds.json"
    "$REPO/target/release/sts-cli" commands < "$FIGHTS" > "$OUT"
    FIGHTS=$OUT
fi
"$PY" - "$CONF" "command=$PY -I $REPO/tools/trace/$SCRIPT $FIGHTS $DELAY" <<'EOF'
import sys
path, line = sys.argv[1], sys.argv[2]
lines = [l for l in open(path).read().splitlines() if not l.startswith("command=")]
open(path, "w").write("\n".join(lines + [line]) + "\n")
EOF
echo "watching $FIGHTS with $SCRIPT (log: $FIGHTS.playback.log)"
cd "$REPO/harness/game"
jre/bin/java -Xmx2G -jar ModTheSpire.jar --skip-launcher --skip-intro --mods basemod,CommunicationMod
