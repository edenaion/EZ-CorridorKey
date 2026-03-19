import timeit
import numpy as np
from CorridorKeyModule.inference_engine_ort import CorridorKeyEngineORT

def process_frame(engine):
    img  = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8) / 255.0
    mask = np.random.randint(0, 255, (2048, 2048),    dtype=np.uint8) / 255.0
    engine.process_frame(img, mask)

def test():
    print("--- ORT SPLIT BENCHMARK (CPU Backbone + DML Refiner) ---")
    engine = CorridorKeyEngineORT(
        backbone_onnx="CorridorKeyModule/checkpoints/CorridorKey_backbone.onnx",
        refiner_onnx="CorridorKeyModule/checkpoints/CorridorKey_refiner.onnx",
        img_size=1024,
    )

    print("[SYSTEM] Running 1 warmup pass (DML compiles shaders on first run)...")
    process_frame(engine)

    iterations = 5
    print(f"[SYSTEM] Running {iterations} timed passes...")
    elapsed = timeit.timeit(lambda: process_frame(engine), number=iterations)
    avg = elapsed / iterations
    print(f"\n--- RESULTS ---")
    print(f"Average Speed: {avg:.3f}s per frame")
    print(f"Approx FPS:    {1/avg:.2f}")

if __name__ == "__main__":
    test()
