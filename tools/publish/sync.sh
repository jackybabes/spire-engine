#!/bin/sh
# Brings the public repository up to date with this one: exports the public part (tools/publish/export.py, which refuses
# anything private), commits whatever changed there and pushes it. Work stays in this repository; run this when you want
# the public one to catch up.
#
#     tools/publish/sync.sh [public repo folder, default ../spire-engine] ["commit message"]
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DEST=${1:-$REPO/../spire-engine}
MSG=${2:-"Sync from the development repository ($(date +%Y-%m-%d))"}
python3 "$REPO/tools/publish/export.py" "$DEST"
cd "$DEST"
git add -A
if git diff --cached --quiet; then
    echo "nothing changed"
    exit 0
fi
git commit -q -m "$MSG"
git push -q
echo "pushed: $(git log --oneline -1)"
