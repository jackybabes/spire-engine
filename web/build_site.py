"""Builds the hosted viewer: the page into web/dist/ (Cloudflare Pages) and the engine and the
networks' weights into web/dist-assets/ (the R2 bucket spire-viewer-assets, which the Pages
Function web/functions/assets/[[path]].js serves at /assets/). web/deploy.sh uploads both.

    python3 web/build_site.py

- the page, its scripts and styles, gallery.json; the page with Cloudflare Web Analytics' beacon
  (page views counted without cookies; the hosted page only, so serving it locally isn't counted);
- the engine (web/build.sh), named by its content (sts.<hash>.wasm), loaded from /assets/;
- each network in networks.json, its weights from web/models/ under a content-hashed name, loaded
  from /assets/, and the engine it was released with if it names one (`engine`, the release's
  sts.wasm): a network made for an older encoding than the current engine's must, or the build
  stops. Hashed names never change content, so they are cached for a year and an old page always
  finds the engine it was built with;
- the gallery's runs (python/results/watch/) under runs/, and runs/index.json listing them
  (web/serve.py makes that list on the fly when serving locally);
- _headers: nosniff (Pages has browsers recheck the page and its runs on every visit).

It refuses to finish if dist/ would hold a file type the site doesn't use, or text that
tools/publish/export.py counts as private (the decompiled game, machine names, addresses, home
folders, your email, the private notes).
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

WEB = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(WEB)
DIST = os.path.join(WEB, "dist")
ASSETS = os.path.join(WEB, "dist-assets")  # uploaded to R2, served at /assets/
RUNS = os.path.join(ROOT, "python", "results", "watch")
PAGE = ["index.html", "app.js", "engine.js", "text.js", "text-cards.js", "text-events.js", "art.js", "style.css", "gallery.json",
        "manifest.webmanifest", "icon-180.png", "icon-192.png", "icon-512.png"]
ALLOWED = {".html", ".js", ".css", ".json", ".wasm", ".safetensors", ".webmanifest", ".png", ""}  # "" is _headers
# Cloudflare Web Analytics (jack, 2026-10-29: visits weren't reported; the site set up on 2026-10-24 counts only
# pages that load this). Its token is the Web Analytics site's public one, in every page it counts, not an API token.
ANALYTICS = ('<script type="module" src="https://static.cloudflareinsights.com/beacon.min.js" '
             "data-cf-beacon='{\"token\": \"8ff8f8bfa0304d5aac3f86bb8d13163e\"}'></script>")


def hashed(src, dest_dir, stem, ext):
    data = open(src, "rb").read()
    name = f"{stem}.{hashlib.sha256(data).hexdigest()[:10]}{ext}"
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, name), "wb") as f:
        f.write(data)
    return name


def weights_metadata(path):
    """A safetensors file's metadata (8 bytes of header length, then the JSON header)."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n)).get("__metadata__", {})


def encoding_version():
    """The current engine's encoding version (crates/sts-encode's ENCODING_VERSION)."""
    src = open(os.path.join(ROOT, "crates", "sts-encode", "src", "encode.rs")).read()
    return int(re.search(r"pub const ENCODING_VERSION: u32 = (\d+);", src).group(1))


def private_checks():
    spec = importlib.util.spec_from_file_location("export", os.path.join(ROOT, "tools", "publish", "export.py"))
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    checks = dict(export.FORBIDDEN)
    email = export.own_email()
    if email:
        checks["your email address"] = email
    return checks


def main():
    subprocess.run(["sh", os.path.join(WEB, "build.sh")], check=True)
    shutil.rmtree(DIST, ignore_errors=True)
    shutil.rmtree(ASSETS, ignore_errors=True)
    os.makedirs(DIST)
    os.makedirs(ASSETS)
    for f in PAGE:
        shutil.copy(os.path.join(WEB, f), os.path.join(DIST, f))
    page = os.path.join(DIST, "index.html")
    src = open(page).read()
    assert src.count("</head>") == 1
    open(page, "w").write(src.replace("</head>", f"  {ANALYTICS}\n</head>"))

    wasm = hashed(os.path.join(WEB, "sts.wasm"), ASSETS, "sts", ".wasm")
    engine = os.path.join(DIST, "engine.js")
    src = open(engine).read()
    assert 'ENGINE = "sts.wasm"' in src
    open(engine, "w").write(src.replace('ENGINE = "sts.wasm"', f'ENGINE = "/assets/{wasm}"'))

    nets = json.load(open(os.path.join(WEB, "networks.json")))
    current = encoding_version()
    for n in nets:
        weights = os.path.join(WEB, n["file"])
        if not os.path.exists(weights):
            sys.exit(f"{n['file']} is missing: gh release download {n['name']} -p {os.path.basename(n['file'])} -D web/models")
        version = int(weights_metadata(weights).get("encoding_version", 0))
        if n.get("engine"):
            own = os.path.join(WEB, n["engine"])
            if not os.path.exists(own):
                sys.exit(f"{n['engine']} is missing: gh release download {n['name']} -p sts.wasm -O web/{n['engine']}")
            n["engine"] = "/assets/" + hashed(own, ASSETS, "sts", ".wasm")
        elif version != current:
            sys.exit(f"{n['name']} was made for encoding {version} and the engine is at {current}: give it the engine it was "
                     f'released with in networks.json ("engine": "models/{n["name"]}.sts.wasm", the release\'s sts.wasm)')
        n["bytes"] = os.path.getsize(weights)  # shown before the download (the page's AI choice)
        n["file"] = "/assets/" + hashed(weights, ASSETS, n["name"], ".safetensors")
    json.dump(nets, open(os.path.join(DIST, "networks.json"), "w"), indent=2)

    runs = [g["file"] for g in json.load(open(os.path.join(WEB, "gallery.json")))]
    os.makedirs(os.path.join(DIST, "runs"))
    for r in runs:
        shutil.copy(os.path.join(RUNS, r), os.path.join(DIST, "runs", r))
    json.dump(sorted(runs), open(os.path.join(DIST, "runs", "index.json"), "w"))

    with open(os.path.join(DIST, "_headers"), "w") as f:
        # No Cache-Control here: Pages already has browsers recheck its files on every visit, and
        # a rule here would also override the year the /assets/ Function sets.
        f.write("/*\n  X-Content-Type-Options: nosniff\n")

    checks, problems, total = private_checks(), [], 0
    walk = [(d, fs) for top in (DIST, ASSETS) for d, _, fs in os.walk(top)]
    for d, files in walk:
        for f in files:
            p = os.path.join(d, f)
            rel = os.path.relpath(p, WEB)
            total += os.path.getsize(p)
            if os.path.splitext(f)[1] not in ALLOWED or re.search(r"(^|/)(reference|harness|notes)(/|$)", rel):
                problems.append(f"{rel}: not a file the site uses")
            if f.endswith((".html", ".js", ".css", ".json", ".webmanifest")):
                text = open(p, encoding="utf-8", errors="replace").read()
                problems += [f"{rel}: {what}" for what, pat in checks.items() if pat.search(text)]
    if problems:
        shutil.rmtree(DIST)
        shutil.rmtree(ASSETS)
        sys.exit("Not built; fix these first:\n  " + "\n  ".join(problems))
    for d, files in sorted(walk):
        for f in sorted(files):
            p = os.path.join(d, f)
            print(f"  {os.path.relpath(p, WEB):52} {os.path.getsize(p) / 1e6:6.2f} MB")
    print(f"web/dist and web/dist-assets: {total / 1e6:.1f} MB, nothing private found")


if __name__ == "__main__":
    main()
