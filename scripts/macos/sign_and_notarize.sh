#!/usr/bin/env bash
# Sign and notarize CorridorKey.app for distribution
#
# Prerequisites:
#   - Apple Developer ID Application certificate in Keychain
#   - Apple Developer ID Installer certificate in Keychain
#   - App-specific password stored in Keychain as "AC_PASSWORD"
#     (create at appleid.apple.com > Security > App-Specific Passwords)
#     xcrun notarytool store-credentials "AC_PASSWORD" \
#       --apple-id "your@email.com" --team-id "TEAMID" --password "xxxx-xxxx-xxxx-xxxx"
#
# Usage: bash scripts/macos/sign_and_notarize.sh
#
# Environment variables (override defaults):
#   SIGN_IDENTITY   - "Developer ID Application: ..." (auto-detected if one exists)
#   INSTALLER_IDENTITY - "Developer ID Installer: ..." (auto-detected)
#   APPLE_ID        - Apple ID email
#   TEAM_ID         - Apple Developer Team ID
#   KEYCHAIN_PROFILE - notarytool credential profile name (default: AC_PASSWORD)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_PATH="$ROOT/dist/EZ-CorridorKey.app"
ENTITLEMENTS="$SCRIPT_DIR/CorridorKey.entitlements"
VERSION=$(python3 -c "import tomllib; print(tomllib.load(open('$ROOT/pyproject.toml','rb'))['project']['version'])" 2>/dev/null)
VERSION="${VERSION:-0.0.0}"
PKG_NAME="EZ-CorridorKey-${VERSION}.pkg"
KEYCHAIN_PROFILE="${KEYCHAIN_PROFILE:-AC_PASSWORD}"

if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: $APP_PATH not found. Run build_macos.sh first."
    exit 1
fi

# --- Auto-detect signing identity ---
if [ -z "${SIGN_IDENTITY:-}" ]; then
    SIGN_IDENTITY=$(security find-identity -v -p codesigning | grep "Developer ID Application" | head -1 | sed 's/.*"\(.*\)".*/\1/' || true)
    if [ -z "$SIGN_IDENTITY" ]; then
        echo "ERROR: No 'Developer ID Application' certificate found in Keychain."
        echo "  Install your certificate or set SIGN_IDENTITY env var."
        exit 1
    fi
fi
echo "Signing with: $SIGN_IDENTITY"

if [ -z "${INSTALLER_IDENTITY:-}" ]; then
    INSTALLER_IDENTITY=$(security find-identity -v | grep "Developer ID Installer" | head -1 | sed 's/.*"\(.*\)".*/\1/' || true)
fi

# --- Step 1: Sign the .app (bottom-up, then deep-sign top level) ---
echo ""
echo "=== Step 1: Code Signing ==="

# Sign inner shared libraries and frameworks first (bottom-up).
# This ensures nested code objects are signed before the top-level
# bundle, which Apple requires for valid signatures.
echo "Signing inner dylibs and .so files..."
find "$APP_PATH/Contents" \( -name "*.dylib" -o -name "*.so" \) -print0 | \
    xargs -0 -n1 codesign --force --options runtime \
        --entitlements "$ENTITLEMENTS" \
        --sign "$SIGN_IDENTITY" 2>&1 || true

# Sign embedded executables (skip data files)
echo "Signing embedded executables..."
find "$APP_PATH/Contents/MacOS" -type f -perm +111 ! -name "*.pth" ! -name "*.pt" \
    ! -name "*.safetensors" ! -name "*.bin" \
    -print0 | \
    xargs -0 -n1 codesign --force --options runtime \
        --entitlements "$ENTITLEMENTS" \
        --sign "$SIGN_IDENTITY" 2>&1 || true

# Deep-sign the top-level .app bundle.
# Must run locally (not over SSH) — requires GUI security agent.
echo "Signing top-level .app (deep)..."
codesign --deep --force --options runtime \
    --entitlements "$ENTITLEMENTS" \
    --sign "$SIGN_IDENTITY" \
    "$APP_PATH"

echo "Verifying signature..."
codesign --verify --deep --strict "$APP_PATH"
echo "Signature OK"

# --- Step 2: Notarize ---
echo ""
echo "=== Step 2: Notarization ==="
ZIP_PATH="$ROOT/dist/EZ-CorridorKey.zip"
echo "Creating zip for notarization..."
ditto -c -k --keepParent "$APP_PATH" "$ZIP_PATH"

echo "Submitting to Apple (this may take a few minutes)..."
xcrun notarytool submit "$ZIP_PATH" \
    --keychain-profile "$KEYCHAIN_PROFILE" \
    --wait

echo "Stapling notarization ticket..."
xcrun stapler staple "$APP_PATH"

# Clean up zip
rm -f "$ZIP_PATH"

echo "Notarization complete"

# --- Step 3: Build .pkg ---
echo ""
echo "=== Step 3: Building .pkg Installer ==="
PKG_PATH="$ROOT/dist/$PKG_NAME"

if [ -n "${INSTALLER_IDENTITY:-}" ]; then
    pkgbuild --component "$APP_PATH" \
        --install-location /Applications \
        --sign "$INSTALLER_IDENTITY" \
        "$PKG_PATH"
else
    echo "WARNING: No 'Developer ID Installer' certificate found."
    echo "  Building unsigned .pkg (won't pass Gatekeeper for distribution)."
    pkgbuild --component "$APP_PATH" \
        --install-location /Applications \
        "$PKG_PATH"
fi

PKG_SIZE=$(du -sh "$PKG_PATH" | awk '{print $1}')

echo ""
echo "=== Done ==="
echo "  Signed app: $APP_PATH"
echo "  Installer:  $PKG_PATH ($PKG_SIZE)"
echo ""
echo "Test install: sudo installer -pkg $PKG_PATH -target /"
echo "Or just double-click $PKG_NAME in Finder."
