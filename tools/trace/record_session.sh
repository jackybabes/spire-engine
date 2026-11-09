#!/bin/sh
# One recording session in the real game, unattended:
#
#     tools/trace/record_session.sh <trace dir> <minutes> <recorder command...>
#
# e.g. tools/trace/record_session.sh harness/traces/f1 20 /opt/homebrew/bin/python3 -I \
#          "$PWD/tools/trace/record.py" "$PWD/harness/traces/f1" 40 12 6 act2 120 --character silent
# Points CommunicationMod at the recorder, runs the modded game until the recorder logs "done"
# (or crashes, or the minutes run out), closes the game and restores the previous config.
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CONF="$HOME/Library/Preferences/ModTheSpire/CommunicationMod/config.properties"
OUT=$1; MINUTES=$2; shift 2
cp "$CONF" "$CONF.before-session"
/opt/homebrew/bin/python3 - "$CONF" "command=$*" <<'PY'
import sys
path, line = sys.argv[1], sys.argv[2]
lines = [l for l in open(path).read().splitlines() if not l.startswith("command=")]
lines.append(line)
open(path, "w").write("\n".join(lines) + "\n")
PY
(cd "$REPO/harness/game" && exec jre/bin/java -Xmx2G -jar ModTheSpire.jar --skip-launcher --skip-intro --mods basemod,CommunicationMod > "$OUT.game.log" 2>&1) &
PID=$!
END=$(( $(date +%s) + MINUTES * 60 ))
while kill -0 $PID 2>/dev/null; do
    if grep -qE " done$| crash " "$OUT/recorder.log" 2>/dev/null; then break; fi
    if [ "$(date +%s)" -ge "$END" ]; then echo "timed out"; break; fi
    sleep 10
done
kill $PID 2>/dev/null
sleep 2
mv "$CONF.before-session" "$CONF"
echo "recorded $(ls "$OUT" | grep -c jsonl) traces; last log lines:"
tail -3 "$OUT/recorder.log"
