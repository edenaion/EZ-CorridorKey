#!/usr/bin/env bash
# Generate corridorkey.icns from corridorkey.svg (transparent background)
# Run on macOS: bash scripts/macos/generate_icns.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SVG="$ROOT/ui/theme/corridorkey.svg"
DEST="$ROOT/ui/theme/corridorkey.icns"
ICONSET="$ROOT/corridorkey.iconset"
MASTER_PNG="$ROOT/corridorkey_master_1024.png"

if [ ! -f "$SVG" ]; then
    echo "ERROR: Source SVG not found: $SVG"
    exit 1
fi

echo "Generating .icns from $SVG (transparent background)..."

# Render SVG to 1024x1024 PNG with transparent background
# Try cairosvg first (pip install cairosvg), then rsvg-convert (brew install librsvg),
# then fall back to qlmanage
if python3 -c "import cairosvg" 2>/dev/null; then
    echo "Using cairosvg..."
    python3 -c "
import cairosvg
cairosvg.svg2png(
    url='$SVG',
    write_to='$MASTER_PNG',
    output_width=1024,
    output_height=1024,
    background_color=None
)
"
elif command -v rsvg-convert >/dev/null 2>&1; then
    echo "Using rsvg-convert..."
    rsvg-convert -w 1024 -h 1024 --background-color none "$SVG" -o "$MASTER_PNG"
else
    echo "Using qlmanage fallback (may add white background)..."
    echo "  For best results: pip install cairosvg OR brew install librsvg"
    qlmanage -t -s 1024 -o "$(dirname "$MASTER_PNG")" "$SVG" 2>/dev/null
    mv "$(dirname "$MASTER_PNG")/$(basename "$SVG").png" "$MASTER_PNG" 2>/dev/null || true
fi

if [ ! -f "$MASTER_PNG" ]; then
    echo "ERROR: Failed to render SVG to PNG"
    exit 1
fi

# Generate iconset from master PNG
mkdir -p "$ICONSET"
sips -z 16 16     "$MASTER_PNG" --out "$ICONSET/icon_16x16.png"      >/dev/null
sips -z 32 32     "$MASTER_PNG" --out "$ICONSET/icon_16x16@2x.png"   >/dev/null
sips -z 32 32     "$MASTER_PNG" --out "$ICONSET/icon_32x32.png"      >/dev/null
sips -z 64 64     "$MASTER_PNG" --out "$ICONSET/icon_32x32@2x.png"   >/dev/null
sips -z 128 128   "$MASTER_PNG" --out "$ICONSET/icon_128x128.png"    >/dev/null
sips -z 256 256   "$MASTER_PNG" --out "$ICONSET/icon_128x128@2x.png" >/dev/null
sips -z 256 256   "$MASTER_PNG" --out "$ICONSET/icon_256x256.png"    >/dev/null
sips -z 512 512   "$MASTER_PNG" --out "$ICONSET/icon_256x256@2x.png" >/dev/null
sips -z 512 512   "$MASTER_PNG" --out "$ICONSET/icon_512x512.png"    >/dev/null
sips -z 1024 1024 "$MASTER_PNG" --out "$ICONSET/icon_512x512@2x.png" >/dev/null

iconutil -c icns "$ICONSET" -o "$DEST"

# Clean up
rm -rf "$ICONSET" "$MASTER_PNG"

echo "Created: $DEST"
ls -la "$DEST"
