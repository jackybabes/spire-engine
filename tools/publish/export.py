"""Copies this (private) repository's public part into the public repository's folder.

    python3 tools/publish/export.py ../slaythespire-public          # copy and check
    python3 tools/publish/export.py ../slaythespire-public --check  # only check what would go

This repository keeps the full history and the internal notes; the public one gets a copy of the
committed files under PUBLIC (nothing untracked, nothing under notes/, no CLAUDE.md), and its own
history. The copy replaces everything in the target folder except its .git. It refuses to write
anything if a file to publish matches FORBIDDEN_PATHS or holds a FORBIDDEN text (the decompiled
game's sources, the harness, private machine names and addresses, references to the notes).
Committing and pushing the public repository: tools/publish/sync.sh does both after the export.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# What goes public: these top-level entries, as committed.
PUBLIC = ["README.md", "LICENSE", "Cargo.toml", "Cargo.lock", ".gitignore", "crates", "python", "web", "tools", "docs"]
# Left out quietly: old experiment logs and scripts (their command lines carry the training machine's paths).
SKIP = re.compile(r"^python/results/.*\.(log|out|sh)$")
# Never published, even if committed under PUBLIC.
FORBIDDEN_PATHS = re.compile(r"(^|/)(reference|harness|notes|sendToDevs)/|\.(class|jar)$|(^|/)CLAUDE\.md$|(^|/)\.claude/")
FORBIDDEN = {
    "the decompiled game's sources": re.compile(r"^package com\.megacrit", re.M),
    "the training machine": re.compile(r"\bmax2\b"),
    "a private address": re.compile(r"\b192\.168\.\d+\.\d+"),
    "a home folder": re.compile(r"/Users/\w+|/home/\w+/"),
    "the private notes": re.compile(r"\bnotes/(HANDOFF|plan|public-release|reports)"),
}
SELF = "tools/publish/export.py"  # holds the patterns above


def own_email():
    # Your git email, looked up here rather than written into a public file.
    email = subprocess.run(["git", "config", "user.email"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return re.compile(re.escape(email.split("@")[0]) + "@") if email else None


def committed(p):
    return subprocess.run(["git", "show", f"HEAD:{p}"], cwd=ROOT, capture_output=True, check=True).stdout


def files():
    out = subprocess.run(["git", "ls-files", "-z", "--", *PUBLIC], cwd=ROOT, capture_output=True, check=True).stdout
    return sorted(f for f in out.decode().split("\0") if f and not SKIP.search(f))


def problems(paths):
    found = []
    checks = dict(FORBIDDEN)
    email = own_email()
    if email:
        checks["your email address"] = email
    for p in paths:
        if FORBIDDEN_PATHS.search(p):
            found.append(f"{p}: not to be published")
            continue
        if p == SELF:
            continue
        data = committed(p)
        if b"\0" in data[:8000]:
            continue  # binary (weights fixture, gzipped runs)
        text = data.decode("utf-8", "replace")
        for what, pat in checks.items():
            m = pat.search(text)
            if m:
                line = text.count("\n", 0, m.start()) + 1
                found.append(f"{p}:{line}: {what} ({m.group(0)!r})")
    return found


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("target", help="the public repository's folder (created if missing)")
    a.add_argument("--check", action="store_true", help="only check, write nothing")
    args = a.parse_args()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", *PUBLIC], cwd=ROOT, capture_output=True, text=True).stdout
    if dirty:
        sys.exit("Commit the changes under the public paths first (the export copies what is committed):\n" + dirty)
    paths = files()
    found = problems(paths)
    if found:
        print("Not exported; fix these first:", *found, sep="\n  ", file=sys.stderr)
        sys.exit(1)
    size = sum(len(committed(p)) for p in paths)
    print(f"{len(paths)} files, {size / 1e6:.1f} MB, nothing private found")
    if args.check:
        return
    target = os.path.abspath(args.target)
    if target == ROOT or target.startswith(ROOT + os.sep):
        sys.exit("the target must be outside this repository")
    os.makedirs(target, exist_ok=True)
    for entry in os.listdir(target):
        if entry != ".git":
            full = os.path.join(target, entry)
            shutil.rmtree(full) if os.path.isdir(full) and not os.path.islink(full) else os.remove(full)
    # The committed version of each file (git show), not the working copy.
    for p in paths:
        dest = os.path.join(target, p)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(committed(p))
        if os.access(os.path.join(ROOT, p), os.X_OK):
            os.chmod(dest, 0o755)
    print(f"exported to {target}")


if __name__ == "__main__":
    main()
