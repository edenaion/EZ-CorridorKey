import torch
import cv2
import numpy as np
from CorridorKeyModule.inference_engine import CorridorKeyEngine

def test_vram():
    print("Loading engine...")
    engine = CorridorKeyEngine(checkpoint_path="CorridorKeyModule/checkpoints/CorridorKey_v1.0.pth", img_size=2048)
    
    # Create dummy data
    img = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
    mask = np.random.randint(0, 255, (2160, 3840), dtype=np.uint8)
    
    # Reset stats
    torch.cuda.reset_peak_memory_stats()
    
    print("Running inference pass...")
    out = engine.process_frame(img, mask)
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Peak VRAM used: {peak_vram:.2f} GB")

if __name__ == "__main__":
    test_vram()
