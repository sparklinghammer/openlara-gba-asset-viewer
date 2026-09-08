#!/usr/bin/env python3
"""Build an asset-viewer ROM entirely from this folder, on any platform.

The OpenLara checkout is a donor: the handful of source files the GBA target
compiles are mirrored into this project's work folder, the viewer's own sources
are laid over them, and everything is built there. Nothing is ever written back
into the OpenLara tree, so that checkout can be read-only, shared, or updated
from upstream without this project interfering.

One driver for Windows, Linux and macOS. Only two things actually differ, and
both are Windows problems: devkitARM's bundled shell cannot cope with spaces in
a path, so each tree is reached through a temporary drive letter mapped past
them; and that shell wants its DEVKITPRO in MSYS form, /c/... rather than
C:\\... . Everywhere else the paths are handed to make as they are.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import string
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parent
WINDOWS = os.name == "nt"

# SOURCES and INCLUDES in the Makefile are ../../fixed, . and asm; nothing else
# is compiled, so nothing else is mirrored. That is about 60 files, against the
# 265 MB the platform folder holds once its WAD staging and packer tools count.
DONOR_SUFFIXES = {".cpp", ".c", ".h", ".s"}
DONOR_SUBFOLDERS = ("asm", "include")


class BuildError(Exception):
    """Something the person running the build can act on."""


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

def read_settings() -> dict:
    path = PROJECT / "settings.json"
    if not path.is_file():
        return {}
    try:
        # utf-8-sig, not utf-8: Notepad and PowerShell both write a byte order
        # mark, and a settings.json edited by hand would otherwise stop the
        # build with a decoder traceback rather than a sentence.
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise BuildError(f"{path} is not valid JSON: {error}") from error


def resolve(value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (PROJECT / candidate)


# ---------------------------------------------------------------------------
# toolchain
# ---------------------------------------------------------------------------

def find_devkitpro(configured: str | None) -> Path:
    """Where devkitARM lives, by the marker file the Makefile itself needs.

    The installers set DEVKITPRO, so that is tried first and is enough for most
    people. A toolchain unpacked somewhere of your own goes in settings.json
    instead of into the environment.
    """
    candidates = [configured, os.environ.get("DEVKITPRO"), "/opt/devkitpro"]
    if WINDOWS:
        candidates.append("C:/devkitPro")
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser()
        if not root.is_absolute():
            root = PROJECT / root
        if (root / "devkitARM" / "gba_rules").is_file():
            return root.resolve()
    raise BuildError(
        "devkitPro/devkitARM not found. Install the gba-dev group -- the "
        "installer sets DEVKITPRO for you -- or put the path in settings.json "
        "as \"devkitpro\", pointing at the folder that holds "
        "devkitARM/gba_rules."
    )


def find_make() -> str:
    for name in ("make", "gmake", "mingw32-make"):
        found = shutil.which(name)
        if found:
            return found
    raise BuildError("GNU make not found on PATH.")


# ---------------------------------------------------------------------------
# mirroring
# ---------------------------------------------------------------------------

def copy_if_newer(source: Path, target: Path) -> bool:
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def copy_if_different(source: Path, target: Path) -> bool:
    """Compared by content, not by date.

    The overlay has to win even when the donor's copy is the fresher file, which
    is exactly what happens the first time you build after updating or restoring
    the checkout.
    """
    if target.exists():
        a = hashlib.sha256(source.read_bytes()).digest()
        b = hashlib.sha256(target.read_bytes()).digest()
        if a == b:
            return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def mirror_donor(openlara: Path, src_root: Path) -> int:
    copied = 0
    fixed = openlara / "src" / "fixed"
    # fixed/ carries lang, fmt and render subfolders that its headers include.
    for file in sorted(p for p in fixed.rglob("*") if p.is_file()):
        if copy_if_newer(file, src_root / "fixed" / file.relative_to(fixed)):
            copied += 1

    donor_gba = openlara / "src" / "platform" / "gba"
    work_gba = src_root / "platform" / "gba"
    for file in sorted(p for p in donor_gba.iterdir() if p.is_file()):
        if file.suffix in DONOR_SUFFIXES or file.name == "Makefile":
            if copy_if_newer(file, work_gba / file.name):
                copied += 1
    for name in DONOR_SUBFOLDERS:
        folder = donor_gba / name
        if not folder.is_dir():
            continue
        for file in sorted(p for p in folder.rglob("*") if p.is_file()):
            if copy_if_newer(file, work_gba / name / file.relative_to(folder)):
                copied += 1
    return copied


def lay_overlay(src_root: Path) -> int:
    """The viewer's sources, and the donor files it needs hooks in.

    Laying them over the mirror is what lets the OpenLara checkout stay pristine.
    """
    overlay = PROJECT / "overlay" / "src"
    copied = 0
    for file in sorted(p for p in overlay.rglob("*") if p.is_file()):
        if copy_if_different(file, src_root / file.relative_to(overlay)):
            copied += 1
    return copied


# ---------------------------------------------------------------------------
# compiling
# ---------------------------------------------------------------------------

def drive_in_use(letter: str) -> bool:
    try:
        return Path(f"{letter}:/").exists()
    except OSError:
        # An empty card reader answers neither yes nor no: it raises. A letter
        # we cannot read is a letter we must not claim.
        return True


def free_drive_letter(reserved: frozenset[str] = frozenset()) -> str:
    for letter in "QRSTUVWYZ":
        if letter not in reserved and not drive_in_use(letter):
            return letter
    raise BuildError("No temporary drive letter available.")


def discard_stale_objects(work_gba: Path, signature: str) -> None:
    """Object files remember the paths they were compiled through.

    The dependency files gcc leaves behind name every header by absolute path,
    and on Windows those paths run through whichever drive letter happened to
    be free that day. Let a build inherit objects made through a different
    mapping and make stops on a header it cannot find, naming a drive that now
    means something else entirely. Recompiling is cheaper than explaining.
    """
    stamp = work_gba / ".toolchain-paths"
    if stamp.is_file() and stamp.read_text(encoding="utf-8") == signature:
        return
    shutil.rmtree(work_gba / "build", ignore_errors=True)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(signature, encoding="utf-8")


def run_make(work_gba: Path, devkitpro: Path, make: str, mode: str, target: str) -> None:
    arguments = [
        make,
        # The Makefile calls itself, and make hands the sub-make its own path.
        # That path is whatever PATH happened to yield -- a POSIX one under a
        # Git Bash prompt, which devkitARM's shell then cannot find. The bare
        # name is resolvable from either side.
        f"MAKE={Path(make).name}",
        "ASSET_VIEWER=1",
        f"ASSET_VIEWER_FULL={1 if mode == 'full' else 0}",
        "DATA=.asset_viewer_pack_data",
        f"TARGET={target}",
        "BUILD=build",
        f"-j{max(1, os.cpu_count() or 1)}",
    ]
    environment = dict(os.environ)
    make_dir = str(Path(make).parent)

    if not WINDOWS:
        environment["DEVKITPRO"] = str(devkitpro)
        environment["DEVKITARM"] = str(devkitpro / "devkitARM")
        environment["PATH"] = os.pathsep.join([
            str(devkitpro / "devkitARM" / "bin"),
            str(devkitpro / "tools" / "bin"),
            make_dir,
            environment.get("PATH", ""),
        ])
        discard_stale_objects(work_gba, f"{work_gba}\n{devkitpro}")
        subprocess.run(arguments, cwd=work_gba, env=environment, check=True)
        return

    # devkitARM's shell splits its arguments on spaces, and a Windows path
    # nearly always has one: "OneDrive - Contoso", "My Documents", a surname.
    # Each tree therefore gets its own temporary drive letter, mapped past the
    # part that carries them. Two letters rather than one shared root: the
    # toolchain is usually in C:\devkitPro and the project under C:\Users\...,
    # which have only the drive itself in common -- and mapping a whole drive
    # would leave every space exactly where it was.
    build_root = work_gba.parents[2]
    build_drive = free_drive_letter()
    devkit_drive = free_drive_letter(frozenset(build_drive))

    # devkitPro is reached through its parent, so DEVKITPRO keeps the
    # /q/devkitPro shape that devkitPro's own rules are written against.
    devkit_root, devkit_name = devkitpro.parent, devkitpro.name
    if devkit_root == devkitpro:  # a toolchain unpacked straight onto a drive
        devkit_root, devkit_name = devkitpro, ""
    devkit_windows = f"{devkit_drive}:\\{devkit_name}".rstrip("\\")
    msys_devkit = f"/{devkit_drive.lower()}/{devkit_name}".rstrip("/")

    work_cwd = f"{build_drive}:\\{work_gba.relative_to(build_root)}"
    discard_stale_objects(work_gba, f"{work_cwd}\n{msys_devkit}")

    mapped = [(build_drive, build_root), (devkit_drive, devkit_root)]
    for letter, folder in mapped:
        subprocess.run(["subst", f"{letter}:", str(folder)], check=True, shell=True)
    try:
        environment["DEVKITPRO"] = msys_devkit
        environment["DEVKITARM"] = f"{msys_devkit}/devkitARM"
        environment["PATH"] = os.pathsep.join([
            f"{devkit_windows}\\devkitARM\\bin",
            f"{devkit_windows}\\tools\\bin",
            make_dir,
            environment.get("PATH", ""),
        ])
        subprocess.run(arguments, cwd=work_cwd, env=environment, check=True)
    finally:
        for letter, _ in mapped:
            subprocess.run(["subst", f"{letter}:", "/D"], check=False, shell=True)


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openlara", help="the donor checkout; overrides settings.json")
    parser.add_argument("--models", help="folder of models; overrides settings.json")
    parser.add_argument("--tr1", action="store_true",
                        help="build the Tomb Raider catalogue instead, from the donor's own data")
    parser.add_argument("--full-campaign", action="store_true",
                        help="the full Tomb Raider catalogue; needs --wad")
    parser.add_argument("--wad", help="a local TR1.WAD, for --full-campaign")
    parser.add_argument("--clean", action="store_true", help="delete the work folder first")
    args = parser.parse_args()

    settings = read_settings()

    openlara = resolve(args.openlara or settings.get("openlara"))
    if openlara is None:
        raise BuildError(
            "No OpenLara checkout configured. Copy settings.example.json to "
            "settings.json and set \"openlara\" to the folder holding "
            "src/platform/gba, or pass --openlara <path>."
        )
    if not openlara.is_dir():
        raise BuildError(f"The configured OpenLara folder does not exist: {openlara}")
    openlara = openlara.resolve()
    donor_gba = openlara / "src" / "platform" / "gba"
    if not (donor_gba / "Makefile").is_file():
        raise BuildError(
            f"That folder is not an OpenLara checkout: {openlara}\n"
            "Expected src/platform/gba/Makefile underneath it."
        )

    models_dir = resolve(args.models or settings.get("models") or "models")
    roms_dir = resolve(settings.get("roms") or "roms")
    work_dir = resolve(settings.get("work") or "build")

    if args.tr1 and args.full_campaign:
        raise BuildError("--tr1 and --full-campaign are different catalogues; pick one.")
    mode = "full" if args.full_campaign else "tr1" if args.tr1 else "custom"

    if mode == "custom":
        if not models_dir.is_dir():
            raise BuildError(
                f"Model folder not found: {models_dir}\nCreate it and drop your "
                ".glb, .gltf, .dae, .obj or .fbx files in, or point \"models\" "
                "in settings.json elsewhere."
            )
        models_dir = models_dir.resolve()
    wad = None
    if mode == "full":
        if not args.wad:
            raise BuildError("A full-campaign build needs --wad with a local TR1.WAD path.")
        wad = Path(args.wad).expanduser()
        if not wad.is_file():
            raise BuildError(f"TR1.WAD not found: {wad}")
        wad = wad.resolve()

    if args.clean and work_dir.exists():
        shutil.rmtree(work_dir)
    roms_dir.mkdir(parents=True, exist_ok=True)

    devkitpro = find_devkitpro(settings.get("devkitpro"))
    make = find_make()

    # ---- mirror ----------------------------------------------------------
    src_root = work_dir / "src"
    work_gba = src_root / "platform" / "gba"
    work_gba.mkdir(parents=True, exist_ok=True)
    (src_root / "fixed").mkdir(parents=True, exist_ok=True)

    copied = mirror_donor(openlara, src_root) + lay_overlay(src_root)
    print(f"Sources: {copied} file(s) refreshed from {openlara}")

    # Toolchain fixes the donor needs to compile here, applied to the copy only.
    subprocess.run([sys.executable, str(SCRIPTS / "prepare_work_tree.py"),
                    "--src", str(src_root)], check=True)

    # ---- the asset pack --------------------------------------------------
    stage = work_gba / ".asset_viewer_pack_data"
    stage.mkdir(parents=True, exist_ok=True)
    for stale in stage.iterdir():
        if stale.is_file() and stale.name != "ASSET_VIEWER.PAK":
            stale.unlink()

    pack = stage / "ASSET_VIEWER.PAK"
    report = roms_dir / f"openlara-asset-viewer-{mode}-pack.json"
    donor_data = donor_gba / "data"
    glyphs = donor_data / "TITLE.PKD"

    if mode == "custom":
        if not glyphs.is_file():
            raise BuildError(
                f"TITLE.PKD not found in the donor: {glyphs}\n"
                "The viewer borrows its font from it."
            )
        pack_args = [str(SCRIPTS / "build_custom_pack.py"),
                     "--output", str(pack), "--report", str(report),
                     "--models", str(models_dir), "--glyphs", str(glyphs)]
    else:
        pack_args = [str(SCRIPTS / "build_asset_pack.py"),
                     "--output", str(pack), "--report", str(report),
                     "--names", str(openlara / "src" / "fixed" / "common.h")]
        if mode == "full":
            pack_args += ["--title", str(glyphs), "--wad", str(wad)]
        else:
            for name in ("TITLE", "GYM", "LEVEL1", "LEVEL2"):
                pack_args += ["--source", f"{name}={donor_data / (name + '.PKD')}"]
    subprocess.run([sys.executable] + pack_args, check=True)

    # ---- compile ---------------------------------------------------------
    target = "OpenLaraAssetViewer"
    run_make(work_gba, devkitpro, make, mode, target)

    built = work_gba / f"{target}.gba"
    if not built.is_file():
        raise BuildError(f"Build succeeded but the ROM is missing: {built}")
    output = roms_dir / f"openlara-asset-viewer-{mode}.gba"
    try:
        shutil.copyfile(built, output)
    except OSError as error:
        # Windows refuses to overwrite a file another process has mapped, and
        # an emulator left open on the last ROM is exactly that. The build
        # itself succeeded, so say where its copy is rather than throwing a
        # traceback that reads as though the model were at fault.
        raise BuildError(
            f"The ROM was built but could not be written to {output}: "
            f"{error.strerror or error}.\n"
            "An emulator usually still has the old one open. Close it and run "
            "the build again.\n"
            f"The build's own copy is ready at {built}."
        ) from error

    data = output.read_bytes()
    print(f"ROM:    {output}")
    print(f"Bytes:  {len(data)}")
    print(f"SHA256: {hashlib.sha256(data).hexdigest().upper()}")
    print(f"Mode:   {mode}")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
    except subprocess.CalledProcessError as error:
        print(f"FAIL: {Path(str(error.cmd[0])).name} exited with {error.returncode}",
              file=sys.stderr)
        raise SystemExit(error.returncode or 1)
