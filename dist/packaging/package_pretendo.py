#!/usr/bin/env python3
"""
Packages PreTenDo Emulator for release into PreTenDo-Emulator-<version>-windows-x64.zip.
Excludes Pretendulator (they are distributed separately).
"""

from pathlib import Path
import hashlib
import shutil
import zipfile
import sys

def package_release(version="0.1.0"):
    repo_root = Path(r"C:\Users\7SiNs\OneDrive\Documents\GitHub\Eglantine")
    work_dir = Path(r"C:\Users\7SiNs\Documents\Codex\2026-11-04\i-ha\work")
    build_bin = work_dir / "perf-build/bin"
    fresh_dir = work_dir / "pretendulator-fresh"
    outputs_dir = Path(r"C:\Users\7SiNs\Documents\Codex\2026-11-04\i-ha\outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    archive_name = f"PreTenDo-Emulator-{version}-windows-x64.zip"

    # Identify pretendo executable
    pretendo_exe = build_bin / "pretendo.exe"
    if not pretendo_exe.exists():
        pretendo_exe = build_bin / "eglantine.exe"
    if not pretendo_exe.exists():
        print(f"Error: Neither pretendo.exe nor eglantine.exe found in {build_bin}", file=sys.stderr)
        return False

    temp_pkg = work_dir / f"pretendo-pkg-temp-{version}"
    if temp_pkg.exists():
        shutil.rmtree(temp_pkg)
    temp_pkg.mkdir(parents=True, exist_ok=True)

    # Copy files from pretendulator-fresh (which has full runtime DLLs and Qt folders)
    excluded_names = {
        "pretendulator.exe", "pretendulator.pdb", "pretendulator-fresh",
        "user", "logos", "outputs"
    }

    for item in fresh_dir.iterdir():
        if item.name.lower() in excluded_names:
            continue
        if item.is_dir():
            shutil.copytree(item, temp_pkg / item.name, dirs_exist_ok=True)
        elif item.is_file():
            # Skip old emulator executable if it exists in fresh_dir
            if item.name.lower() in {"eglantine.exe", "pretendo.exe", "pretendo-switch2.exe"}:
                continue
            shutil.copy2(item, temp_pkg / item.name)

    # Place the newly built pretendo.exe into package as pretendo.exe
    shutil.copy2(pretendo_exe, temp_pkg / "pretendo.exe")

    # Also copy nx2_switches.txt and LICENSE.txt if present
    for doc in ["nx2_switches.txt", "LICENSE.txt", "README.md"]:
        for src in [repo_root / "dist" / doc, repo_root / doc, build_bin / doc]:
            if src.exists() and not (temp_pkg / doc).exists():
                shutil.copy2(src, temp_pkg / doc)
                break

    # Build the zip outside the repository: release files are never committed.
    primary_zip = outputs_dir / archive_name
    with zipfile.ZipFile(primary_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file_path in sorted(temp_pkg.rglob("*")):
            if file_path.is_file():
                rel_path = file_path.relative_to(temp_pkg)
                z.write(file_path, rel_path.as_posix())

    digest = hashlib.sha256(primary_zip.read_bytes()).hexdigest()
    (outputs_dir / f"{archive_name}.sha256.txt").write_text(f"{digest}  {archive_name}\n",
                                                           encoding="utf-8")

    # Cleanup temp
    shutil.rmtree(temp_pkg)

    print(f"Packaged {primary_zip} ({primary_zip.stat().st_size:,} bytes)")
    print(f"SHA256: {digest}")

    return True

if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 else "0.1.0"
    success = package_release(ver)
    sys.exit(0 if success else 1)
