#!/bin/bash
# Corridor Key Launcher - Local Linux/macOS

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOCAL_PYTHON="python3"
LOCAL_SCRIPT="$SCRIPT_DIR/clip_manager.py"

# SAFETY CHECK: Ensure a folder was provided as an argument
if [ -z "$1" ]; then
    echo "[ERROR] No target folder provided."
    echo ""
    echo "USAGE:"
    echo "You can either run this script from the terminal and provide a path:"
    echo "  ./CorridorKey_local.sh /path/to/your/clip/folder"
    echo ""
    echo "Or, in many Linux/macOS desktop environments, you can simply"
    echo "DRAG AND DROP a folder onto this script icon to process it."
    echo ""
    read -p "Press enter to exit..."
    exit 1
fi

# Folder dragged or provided via CLI? Use it as the target path.
TARGET_PATH="$1"

# Strip trailing slash if present
TARGET_PATH="${TARGET_PATH%/}"

echo "Starting Corridor Key locally..."
echo "Target: $TARGET_PATH"

# Run the python script with the target path. 
# We use --win_path as the argument name, but clip_manager.py handles Unix paths fine.
"$LOCAL_PYTHON" "$LOCAL_SCRIPT" --action wizard --win_path "$TARGET_PATH"

read -p "Press enter to close..."
