#!/usr/bin/env sh
#
# Build the asset viewer ROM from the models in models/, on Linux or macOS.
# The Windows equivalent is BUILD_ROM.bat; both call the same Python driver.
#
#   ./build_rom.sh                  build models/
#   ./build_rom.sh path/to/models   build another folder without touching models/
#
# The model list lives in models/pack.json, the path to your OpenLara checkout
# in settings.json. That checkout is only ever read: everything produced by this
# build stays here.

set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

python=$(command -v python3 || command -v python || true)
if [ -z "$python" ]; then
    echo "Python 3 is not on PATH. The converter and the driver both need it." >&2
    echo "Install it, then: pip install pillow" >&2
    exit 1
fi

if [ ! -f "$here/settings.json" ]; then
    echo "settings.json is missing. It says where your OpenLara checkout is." >&2
    echo >&2
    echo "  cp settings.example.json settings.json" >&2
    echo >&2
    echo "then set \"openlara\" to the folder that holds src/platform/gba." >&2
    exit 1
fi

if [ "$#" -gt 0 ]; then
    set -- --models "$@"
fi

"$python" "$here/scripts/build_rom.py" "$@"

echo
echo "Verifying the ROM..."
"$python" "$here/scripts/verify_asset_viewer.py" --bundle "$here"

echo
echo "In the viewer: L / R change model, Start changes animation,"
echo "Select opens the settings menu."
