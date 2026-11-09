#!/bin/sh
# Builds and deploys the hosted viewer (https://spire-viewer.pages.dev): the engine and weights
# to the R2 bucket spire-viewer-assets, then the page to Cloudflare Pages. Run by hand or by
# .github/workflows/deploy-viewer.yml. Needs the weights in web/models/ (gh release download)
# and wrangler signed in (or CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID set).
set -e
cd "$(dirname "$0")"
python3 build_site.py
for f in dist-assets/*; do
  case "$f" in
    *.wasm) type=application/wasm ;;
    *) type=application/octet-stream ;;
  esac
  # Hashed names: an upload of a name already there writes the same bytes again.
  npx --yes wrangler@4 r2 object put "spire-viewer-assets/$(basename "$f")" --file "$f" --content-type "$type" --remote
done
npx --yes wrangler@4 pages deploy --branch "${DEPLOY_BRANCH:-main}" --commit-dirty=true
