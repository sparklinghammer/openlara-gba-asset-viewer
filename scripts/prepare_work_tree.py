#!/usr/bin/env python3
"""Apply the donor compatibility fixes the GBA target needs, to the work tree only.

The OpenLara checkout is read, never written, so anything it needs in order to
compile here is fixed on the mirrored copy instead. These are toolchain fixes,
not viewer features -- they would be needed to build the plain game with this
devkitARM as well.

Each fix is idempotent and matched on a small, distinctive fragment rather than
on a whole file, so a donor that has moved on upstream keeps its changes: only
the few lines that actually conflict are touched. A fix that no longer matches
anything is reported rather than passed over, because that means the donor has
changed underneath it and someone should look.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


class Fix:
    def __init__(self, relative: str, before: str, after: str, why: str, guard: str):
        self.relative = relative
        self.before = before
        self.after = after
        self.why = why
        # What must appear just above `before` for the fix to be unnecessary. A
        # donor that already carries the fix by hand -- this one did -- would
        # otherwise be patched a second time, nesting one guard inside another.
        self.guard = guard

    def apply(self, root: Path) -> str:
        target = root / self.relative
        if not target.is_file():
            return f"missing: {self.relative}"
        text = target.read_text(encoding="utf-8")
        position = text.find(self.before)
        if position < 0:
            return "DOES NOT MATCH"
        if self.guard in text[max(0, position - 200):position]:
            return "already applied"
        target.write_text(text.replace(self.before, self.after, 1),
                          encoding="utf-8", newline="")
        return "applied"


FIXES = [
    Fix(
        "fixed/common.h",
        "inline void* operator new(size_t, void *ptr)\n"
        "{\n"
        "    return ptr;\n"
        "}\n",
        "#if !defined(__GBA__)\n"
        "// devkitARM's headers already declare placement new, and defining it a\n"
        "// second time is an error rather than a duplicate.\n"
        "inline void* operator new(size_t, void *ptr)\n"
        "{\n"
        "    return ptr;\n"
        "}\n"
        "#endif\n",
        "placement new is already provided by devkitARM",
        "__GBA__",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, type=Path,
                        help="the work tree's src folder, e.g. build/src")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = args.src.resolve()
    if not root.is_dir():
        print(f"FAIL: work tree not found: {root}", file=sys.stderr)
        return 1

    failed = False
    for fix in FIXES:
        outcome = fix.apply(root)
        if outcome == "DOES NOT MATCH":
            print(f"FAIL: {fix.relative}: the donor no longer contains the text this fix "
                  f"expects ({fix.why}). It may have been fixed upstream, in which case "
                  f"drop the fix; check before building.", file=sys.stderr)
            failed = True
        elif not args.quiet and outcome == "applied":
            print(f"  donor fix: {fix.relative} -- {fix.why}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
