# VideoMaMa Inference Module

This module provides a standalone interface for running VideoMaMa inference.

## Usage

```python
import sys
# Ensure the parent directory of this module is in sys.path
sys.path.append("/path/to/parent/directory")

from VideoMaMa_Inference_Module import load_videomama_model, run_inference, extract_frames_from_video, save_video

# 1. Load Model
# By default, it loads checkpoints from the local 'checkpoints/' directory inside the module.
# Ensure you have copied 'stable-video-diffusion-img2vid-xt' and 'VideoMaMa' into 'checkpoints/'.
pipeline = load_videomama_model(device="cuda")

# Alternatively, specify custom paths:
# pipeline = load_videomama_model(base_model_path="/path/to/base", unet_checkpoint_path="/path/to/unet", device="cuda")

# 2. Prepare Inputs
# You need a list of RGB frames and a list of mask frames (grayscale)
# Helper function to extract from video:
video_path = "input_video.mp4"
input_frames, fps = extract_frames_from_video(video_path, max_frames=24)

# Load your masks (e.g. from file or other process)
# masks = [ ... list of numpy arrays ... ]
# Ensure len(masks) == len(input_frames)

# 3. Run Inference
output_frames = run_inference(pipeline, input_frames, masks)

# 4. Save Output
save_video(output_frames, "output.mp4", fps)
```

## Requirements

Install dependencies listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```
