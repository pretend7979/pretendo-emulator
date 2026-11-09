#!/usr/bin/env python3
"""
Publishes a PreTenDo Emulator release on GitHub: pushes the branch to main, tags it, makes the
release build (whose version is the tag, so its update check works), packages it and creates the
GitHub release with the zip. Uses your git and gh logins.

usage:
  python dist/packaging/release_pretendo.py 0.1.0 [--commit "message"] [--notes "text"]
                                                          [--dry-run]

  --commit   commit every change first with this message (an Assisted-by footer is added, as
             docs/policies/AI.md asks); without it, everything must already be committed
  --add-footer  add the Assisted-by footer to the last commit (already pushed: main is updated
             with --force-with-lease)
  --fresh-history  start the GitHub repository over: one commit of the current code on a new
             local branch (pretendo-main), force-pushed as main, and every old release and tag
             there deleted. Running it again starts over again. Other local branches are kept.
  --notes    release notes (default: "PreTenDo Emulator <version>")
  --dry-run  print what would run, change nothing
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_pretendo import package_release  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BUILD = Path(r"C:\Users\7SiNs\Documents\Codex\2026-11-04\i-ha\work\perf-build")
OUTPUTS = Path(r"C:\Users\7SiNs\Documents\Codex\2026-11-04\i-ha\outputs")
TOOLCHAIN_BIN = r"C:\msys64\clang64\bin"
GITHUB_REPO = "pretend7979/pretendo-emulator"
REMOTE = "pretendo"
TARGET = "pretendo.exe"  # the emulator's executable (its CMake target is still called yuzu)
ASSISTED_BY = "Assisted-by: Claude Opus 5.5 (code changes and review)"
# Where --fresh-history starts the repository over (no parents, no inherited history).
FRESH_BRANCH = "pretendo-main"

ENV = dict(os.environ, PATH=TOOLCHAIN_BIN + os.pathsep + os.environ.get("PATH", ""))
DRY = False


def run(*args, capture=False, check=True, cwd=REPO, changes=True):
    """Runs a command; with --dry-run, commands that change something are only printed."""
    print("  >", " ".join(str(a) for a in args))
    if DRY and changes:
        return ""
    result = subprocess.run([str(a) for a in args], cwd=cwd, env=ENV, text=True,
                            capture_output=capture)
    if check and result.returncode != 0:
        if capture:
            print(result.stdout + result.stderr)
        sys.exit(f"Failed: {' '.join(str(a) for a in args)}")
    return result.stdout.strip() if capture else ""


def git(*args, **kwargs):
    return run("git", *args, **kwargs)


def main():
    global DRY
    parser = argparse.ArgumentParser(description="Publish a PreTenDo Emulator release")
    parser.add_argument("version", help="e.g. 0.1.0")
    parser.add_argument("--commit", help="commit all changes first with this message")
    parser.add_argument("--add-footer", action="store_true",
                        help="add the Assisted-by footer to the last commit")
    parser.add_argument("--fresh-history", action="store_true",
                        help="replace the GitHub repository's history with one commit of the "
                             "current code (on a new local branch, pretendo-main)")
    parser.add_argument("--notes", help="release notes")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    DRY = options.dry_run
    version = options.version.lstrip("v")
    tag = f"pretendo-v{version}"
    notes = options.notes or f"PreTenDo Emulator {version}"

    print("Checking gh and the repository")
    run("gh", "auth", "status", capture=True, changes=False)
    if git("remote", capture=True, changes=False).split().count(REMOTE) == 0:
        git("remote", "add", REMOTE, f"https://github.com/{GITHUB_REPO}.git")

    rewritten = False
    if options.fresh_history:
        # The repository starts over from today's code: one commit, no inherited history.
        # Run again, it starts over again: always one commit.
        print(f"Starting a new history on the branch {FRESH_BRANCH}")
        git("checkout", "--orphan", f"{FRESH_BRANCH}-new")
        git("add", "-A")
        git("commit", "-m", options.commit or f"PreTenDo Emulator {version}", "-m", ASSISTED_BY)
        git("branch", "-M", FRESH_BRANCH)
        for old_tag in git("tag", "--list", "pretendo-v*", capture=True, changes=False).split():
            git("tag", "-d", old_tag)
        rewritten = True
    elif options.commit:
        if git("status", "--porcelain", capture=True, changes=False):
            git("add", "-A")
            git("commit", "-m", options.commit, "-m", ASSISTED_BY)
    elif git("status", "--porcelain", capture=True, changes=False):
        sys.exit("There are uncommitted changes: commit them, or pass --commit \"message\".")
    message = git("log", "-1", "--format=%B", capture=True, changes=False)
    if options.fresh_history:
        pass
    elif options.add_footer and "Assisted-by:" not in message:
        git("commit", "--amend", "-m", message.strip(), "-m", ASSISTED_BY)
        rewritten = True
    elif "Assisted-by:" not in message and not options.commit:
        print("  Note: the last commit has no Assisted-by footer (docs/policies/AI.md asks for "
              "one); add it with --add-footer")

    # A fresh history has just dropped the local tags (only printed in a dry run).
    existing = "" if options.fresh_history else git("tag", "--list", tag, capture=True, changes=False)
    head = git("rev-parse", "HEAD", capture=True, changes=False)
    if existing and git("rev-list", "-n1", tag, capture=True, changes=False) != head:
        sys.exit(f"Tag {tag} already exists on another commit; use a new version number.")

    print(f"Pushing the source to {GITHUB_REPO} main")
    git("fetch", REMOTE, "main", check=False)
    if options.fresh_history:
        # Every earlier release belongs to the old history.
        for old in run("gh", "release", "list", "--repo", GITHUB_REPO, "--json", "tagName",
                       "--jq", ".[].tagName", capture=True, changes=False).split():
            run("gh", "release", "delete", old, "--repo", GITHUB_REPO, "--yes")
        # Replaces whatever main held, inherited history included.
        git("push", "--force", REMOTE, "HEAD:main")
        # And every tag there: they belong to the old history.
        refs = [line.split()[-1] for line in
                git("ls-remote", "--tags", REMOTE, capture=True, changes=False).splitlines()]
        refs = [ref for ref in refs if not ref.endswith("^{}")]
        for start in range(0, len(refs), 100):
            git("push", REMOTE, "--delete", *refs[start:start + 100])
    elif rewritten:
        # The amended commit replaces the one already on main.
        git("push", f"--force-with-lease=main:{REMOTE}/main", REMOTE, "HEAD:main")
    else:
        git("push", REMOTE, "HEAD:main")
    # This repository's tags are pretendo-v*; a plain v<version> there is one of upstream Eden's,
    # pushed by mistake.
    if not options.fresh_history and git("ls-remote", "--tags", REMOTE, f"refs/tags/v{version}",
                                         capture=True, changes=False):
        git("push", REMOTE, "--delete", f"refs/tags/v{version}")
    print(f"Tagging {tag}")
    if not existing:
        git("tag", "-a", tag, "-m", f"PreTenDo Emulator {version}")
    git("push", REMOTE, tag)

    print("Making the release build")
    configure = run("cmake", "-S", REPO, "-B", BUILD, "-DGIT_RELEASE=1", capture=True)
    reported = re.search(r"\[GetSCMRev\] Git tag: (\S+)", configure or "")
    if not DRY and (not reported or reported.group(1) != tag):
        sys.exit(f"The build would report version {reported.group(1) if reported else '?'}, "
                 f"not {tag}: is another tag on this commit?")
    try:
        # Ninja doesn't recompile the GUI's sources when a .ui file's generated header changes,
        # which leaves old dialog text in the program: regenerate the headers, then drop every GUI
        # object older than the newest one.
        run("cmake", "--build", BUILD, "--target", "yuzu_autogen")
        if not DRY:
            headers = list((BUILD / "src/yuzu/yuzu_autogen/include").glob("ui_*.h"))
            newest = max((h.stat().st_mtime for h in headers), default=0)
            stale = [o for o in (BUILD / "src/yuzu/CMakeFiles/yuzu.dir").rglob("*.obj")
                     if o.stat().st_mtime < newest]
            for obj in stale:
                obj.unlink()
            print(f"  {len(stale)} GUI objects older than the dialogs rebuilt")
        run("cmake", "--build", BUILD, "--target", TARGET)
        print("Packaging")
        if not DRY and not package_release(version):
            sys.exit("Packaging failed")
    finally:
        # Back to a development build, so test builds don't call themselves the release.
        print("Restoring the development build settings")
        run("cmake", "-U", "GIT_RELEASE", "-S", REPO, "-B", BUILD, capture=True)

    archive = OUTPUTS / f"PreTenDo-Emulator-{version}-windows-x64.zip"
    print(f"Publishing the release {tag}")
    exists = run("gh", "release", "view", tag, "--repo", GITHUB_REPO, capture=True, check=False,
                 changes=False)
    if exists:
        run("gh", "release", "upload", tag, archive, f"{archive}.sha256.txt", "--repo",
            GITHUB_REPO, "--clobber")
    else:
        run("gh", "release", "create", tag, archive, f"{archive}.sha256.txt", "--repo",
            GITHUB_REPO, "--title", f"PreTenDo Emulator {version}", "--notes", notes)
    print(f"Done: https://github.com/{GITHUB_REPO}/releases/tag/{tag}")


if __name__ == "__main__":
    main()
