#!/usr/bin/env bash
# Builds ModTheSpire, BaseMod and the patched CommunicationMod from source and assembles an
# isolated game folder in harness/game (git-ignored). Needs git, Maven and a JDK 21.
#
# Run the game afterwards with:
#   cd harness/game && jre/bin/java -Xmx2G -jar ModTheSpire.jar --skip-launcher --skip-intro --mods basemod,CommunicationMod
# and point CommunicationMod at the recorder in
#   ~/Library/Preferences/ModTheSpire/CommunicationMod/config.properties
#   command=/opt/homebrew/bin/python3 -I <repo>/tools/trace/record.py <out dir> <fights> <policy seed> <speed>
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TRACE="$REPO/tools/trace"
SRC="$REPO/harness/src"
GAME="$REPO/harness/game"
STS="${STS_RESOURCES:-/Applications/SlayTheSpire.app/Contents/Resources}"
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@21}"

mkdir -p "$SRC/lib" "$GAME/mods"
ln -sf "$STS/desktop-1.0.jar" "$SRC/lib/desktop-1.0.jar"

clone() {
  local repo=$1 name=$2
  if [ ! -d "$SRC/$name" ]; then
    git clone -q "https://github.com/$repo.git" "$SRC/$name"
  fi
  git -C "$SRC/$name" checkout -q "$(cat "$TRACE/$name.commit")"
}

clone kiooeht/ModTheSpire ModTheSpire
clone daviscook477/BaseMod BaseMod
clone ForgottenArbiter/CommunicationMod CommunicationMod

# ModTheSpire uses sun.misc internals, so it compiles against the game's Java 8 rt.jar.
(cd "$SRC/ModTheSpire" && git apply --check "$TRACE/ModTheSpire.patch" 2>/dev/null && git apply "$TRACE/ModTheSpire.patch" || true)
(cd "$SRC/ModTheSpire" && mvn -q -B package -DskipTests)
cp "$SRC/ModTheSpire/target/ModTheSpire.jar" "$SRC/lib/"

(cd "$SRC/BaseMod" && mvn -q -B package -DskipTests -pl mod -am)
cp "$SRC/BaseMod/mod/target/BaseMod.jar" "$SRC/lib/"

# Local CommunicationMod changes: `harness` command, extra state, speed-up, no music.
(cd "$SRC/CommunicationMod" && git apply --check "$TRACE/CommunicationMod.patch" 2>/dev/null && git apply "$TRACE/CommunicationMod.patch" || true)
(cd "$SRC/CommunicationMod" && mvn -q -B package -DskipTests)

ln -sf "$STS/desktop-1.0.jar" "$GAME/desktop-1.0.jar"
ln -sfn "$STS/jre" "$GAME/jre"
cp "$SRC/ModTheSpire/target/ModTheSpire.jar" "$GAME/"
cp "$SRC/BaseMod/mod/target/BaseMod.jar" "$SRC/CommunicationMod/target/CommunicationMod.jar" "$GAME/mods/"
echo "game folder ready: $GAME"
