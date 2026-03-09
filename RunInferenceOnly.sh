#!/bin/bash

# Ensure script stops on error
set -e

# Path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate Environment
source "${SCRIPT_DIR}/venv/bin/activate"

# Enable OpenEXR Support
export OPENCV_IO_ENABLE_OPENEXR=1

echo "Starting CorridorKey Inference..."
echo "Scanning ClipsForInference for Ready Clips (Input + Alpha)..."

# Run Manager
python "${SCRIPT_DIR}/clip_manager.py" --action run_inference

echo "Inference Complete."
