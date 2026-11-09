#!/bin/sh
# Overnight real-game check on the Mac: plays back a batch of engine runs (python/watch/soak_runs.py)
# in the modded game and logs every run where the game and the engine differ (see soak.py).
#
#     tools/trace/soak.sh [batch.json.gz] [out dir] [delay seconds]
#
# Defaults: python/results/soak/batch.json.gz, harness/soak/<date-time>, 0.02 s between commands
# (0.1 s with SOAK_TICKS=1).
# The game runs 200 updates a drawn frame (SOAK_TICKS, HarnessTicksPatch; CLAUDE.md): every update a
# normal 1/60 s, so the game goes through the same states as at normal speed, about 30 times as
# fast as the old speed 6 (SOAK_TICKS=1 is the old harness: speed 6, frame-time steps).
# The game commands come from the current engine (sts-cli commands, into <out dir>/batch.json),
# not from the batch, so engine fixes to them apply to old batches too.
# SOAK_INSTANCES=N runs N game copies at once (each in harness/game-<i>, playing every Nth run).
# Keeps the Mac awake, restarts the game if it exits or stops responding for 10 minutes (soak.py
# carries on from the next unplayed run; pass the same out dir to resume an interrupted soak),
# and puts CommunicationMod's previous command back at the end. Then:
#
#     python3 tools/trace/soak.py --report <out dir>
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
BATCH=${1:-$REPO/python/results/soak/batch.json.gz}
BATCH=$(cd "$(dirname "$BATCH")" && pwd)/$(basename "$BATCH")
OUT=${2:-$REPO/harness/soak/$(date +%Y%m%d-%H%M)}
TICKS=${SOAK_TICKS:-200}
if [ "$TICKS" -gt 1 ]; then DEFAULT_DELAY=0.02; else DEFAULT_DELAY=0.1; fi
DELAY=${3:-$DEFAULT_DELAY}
# A game that sends no message for this long is restarted (it has to start up, slower with several
# copies swapping), and the copies are checked this often: 600 and 30 s at the old speed.
if [ "$TICKS" -gt 1 ]; then FREEZE_SECONDS=${SOAK_FREEZE:-180}; POLL=10; else FREEZE_SECONDS=${SOAK_FREEZE:-600}; POLL=30; fi
CONF="$HOME/Library/Preferences/ModTheSpire/CommunicationMod/config.properties"
PY=$(command -v /opt/homebrew/bin/python3 || command -v python3)
MAX_LAUNCHES=30

[ -f "$BATCH" ] || { echo "no batch at $BATCH (python/watch/soak_runs.py makes one)"; exit 1; }
[ -f "$CONF" ] || { echo "no CommunicationMod config at $CONF"; exit 1; }
[ -d "$REPO/harness/game" ] || { echo "no harness at $REPO/harness/game (tools/trace/setup.sh builds it)"; exit 1; }
if [ "$TICKS" -gt 1 ] && ! unzip -l "$REPO/harness/game/mods/CommunicationMod.jar" | grep -q HarnessTicksPatch; then
    echo "the harness's CommunicationMod has no ticks (HarnessTicksPatch): rebuild it (tools/trace/setup.sh), or SOAK_TICKS=1"
    exit 1
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

# The game commands come from the current engine, not the batch (whose commands date from when it
# was made): sts-cli commands works them out again from each run's seed and decisions. A resumed
# soak keeps the batch it started with.
if [ ! -f "$OUT/batch.json" ]; then
    (cd "$REPO" && cargo build --release -q -p sts-cli) || { echo "sts-cli doesn't build"; exit 1; }
    case "$BATCH" in
        *.gz) gzip -dc "$BATCH" ;;
        *) cat "$BATCH" ;;
    esac | "$REPO/target/release/sts-cli" commands > "$OUT/batch.json.tmp" || { echo "sts-cli commands failed"; exit 1; }
    mv "$OUT/batch.json.tmp" "$OUT/batch.json"
fi
BATCH=$OUT/batch.json

cp "$CONF" "$CONF.before-soak"
caffeinate -dims &
CAFFEINE=$!
cleanup() {
    for f in "$OUT"/java-*.pid; do
        [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null || true
        rm -f "$f"
    done
    [ -n "$INSTANCE_PIDS" ] && kill $INSTANCE_PIDS 2>/dev/null || true
    kill "$CAFFEINE" 2>/dev/null || true
    mv "$CONF.before-soak" "$CONF"
    echo "restored CommunicationMod config"
    "$PY" "$REPO/tools/trace/soak.py" --report "$OUT" || true
}
trap cleanup EXIT
trap 'exit 1' INT TERM

"$PY" - "$CONF" "command=$PY -I $REPO/tools/trace/soak.py $BATCH $OUT $DELAY" <<'PYEOF'
import sys
path, line = sys.argv[1], sys.argv[2]
lines = [l for l in open(path).read().splitlines() if not l.startswith("command=")]
open(path, "w").write("\n".join(lines + [line]) + "\n")
PYEOF

age() { "$PY" -c 'import os, sys, time; print(int(time.time() - os.path.getmtime(sys.argv[1])))' "$1"; }

# One game, in game folder $2, playing into $3 (copy $1 of $INSTANCES): launch it, restart it if it
# exits or stops sending messages, until its share of the batch is played.
run_instance() {
    i=$1; game=$2; dir=$3
    mkdir -p "$dir"
    launch=0
    while [ ! -f "$dir/done" ] && [ $launch -lt $MAX_LAUNCHES ]; do
        launch=$((launch + 1))
        echo "$(date '+%H:%M:%S') [copy $i] launching the game ($launch)"
        touch "$dir/heartbeat"
        # (soak.py, started by the game, reads HARNESS_TICKS and the shard from this environment)
        (cd "$game" && HARNESS_TICKS=$TICKS SOAK_SHARD=$i SOAK_SHARDS=$INSTANCES exec jre/bin/java -Xmx2G -jar ModTheSpire.jar --skip-launcher --skip-intro --mods basemod,CommunicationMod) &
        java=$!
        echo "$java" > "$OUT/java-$i.pid"
        while kill -0 "$java" 2>/dev/null; do
            sleep $POLL
            if [ -f "$dir/done" ]; then
                echo "$(date '+%H:%M:%S') [copy $i] all its runs played; closing the game"
                kill "$java" 2>/dev/null || true
                break
            fi
            if [ "$(age "$dir/heartbeat")" -gt $FREEZE_SECONDS ]; then
                echo "$(date '+%H:%M:%S') [copy $i] no message from the game for $FREEZE_SECONDS s; restarting it"
                kill "$java" 2>/dev/null || true
                sleep 5
                kill -9 "$java" 2>/dev/null || true
            fi
        done
        wait "$java" 2>/dev/null || true
        rm -f "$OUT/java-$i.pid"
        tail -n 3 "$dir/soak.log" 2>/dev/null | sed "s/^/[copy $i] /" || true
    done
}

# SOAK_INSTANCES > 1 runs that many game copies at once, each in its own copy of harness/game (the
# game keeps its preferences and saves in its folder), playing every SOAK_INSTANCES-th run of the
# batch into <out dir>/shard-<i>; soak.py --report <out dir> combines them.
INSTANCES=${SOAK_INSTANCES:-1}
echo "soak: $BATCH -> $OUT ($INSTANCES game copies, $TICKS updates a frame, $DELAY s between commands)"
INSTANCE_PIDS=
if [ "$INSTANCES" -le 1 ]; then
    run_instance 0 "$REPO/harness/game" "$OUT"
else
    i=0
    while [ $i -lt "$INSTANCES" ]; do
        game="$REPO/harness/game-$i"
        if [ ! -d "$game" ]; then
            rsync -a --exclude 'saves' --exclude '*.log' --exclude 'hs_err_*' --exclude 'sendToDevs' "$REPO/harness/game/" "$game/"
        fi
        # The copy keeps its own preferences and saves, but plays with the harness's current mods.
        rsync -a --delete "$REPO/harness/game/mods/" "$game/mods/"
        run_instance $i "$game" "$OUT/shard-$i" &
        INSTANCE_PIDS="$INSTANCE_PIDS $!"
        sleep 20  # stagger the launches
        i=$((i + 1))
    done
    wait $INSTANCE_PIDS
fi
