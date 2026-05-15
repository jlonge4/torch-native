#!/usr/bin/env bash
# Apply wan22_neuron_patches to the installed Wan 2.2 package.
# Run from the runway-ml root on trn2-2.
set -euo pipefail

PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
WAN_ROOT="${WAN_ROOT:-/home/ubuntu/Wan2.2/wan}"

echo "Patch source : $PATCH_DIR"
echo "Wan package  : $WAN_ROOT"

if [[ ! -d "$WAN_ROOT" ]]; then
    echo "ERROR: $WAN_ROOT not found. Set WAN_ROOT to the wan package directory."
    exit 1
fi

# Back up originals (skip if backup already exists)
backup() {
    local src="$1"
    [[ -f "${src}.orig" ]] || cp "$src" "${src}.orig"
}

backup "$WAN_ROOT/modules/vae2_2.py"
backup "$WAN_ROOT/modules/t5.py"
backup "$WAN_ROOT/modules/model.py"
backup "$WAN_ROOT/textimage2video.py"

cp "$PATCH_DIR/vae2_2.py"        "$WAN_ROOT/modules/vae2_2.py"
cp "$PATCH_DIR/t5.py"            "$WAN_ROOT/modules/t5.py"
cp "$PATCH_DIR/model.py"         "$WAN_ROOT/modules/model.py"
cp "$PATCH_DIR/textimage2video.py" "$WAN_ROOT/textimage2video.py"

echo "Patches applied."
