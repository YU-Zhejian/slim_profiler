import logging
import os

import torch
import time
import argparse
import sys


def gpu_stress_test(device_id:int, duration: int):
    if not torch.cuda.is_available():
        raise ValueError("Error: CUDA not detected.")

    if device_id >= torch.cuda.device_count():
        raise ValueError(f"Error: GPU index {device_id} not found.")

    device = torch.device(f"cuda:{device_id}")
    print(f"--- Target: {torch.cuda.get_device_name(device_id)} (ID: {device_id}) ---")

    # 1. Allocation (~5GB)
    # 1.25 billion float32 elements * 4 bytes = 5GB
    print("Allocating ~5GB VRAM...")
    dummy_memory = torch.zeros((1250000000,), device=device)

    # 2. Setup Calculation (4096 x 4096 matrix)
    N = 4096
    matrix_a = torch.randn((N, N), device=device)
    matrix_b = torch.randn((N, N), device=device)

    # Operations per matmul: 2 * N^3
    ops_per_matmul = 2 * (N**3)

    print(f"Starting stress test for {duration}s...")

    # Synchronization ensures the GPU is ready before the timer starts
    torch.cuda.synchronize(device)
    start_time = time.time()
    iterations = 0

    try:
        while time.time() - start_time < duration:
            # Core calculation
            torch.matmul(matrix_a, matrix_b)
            iterations += 1

        # Wait for the final operations to finish before stopping the clock
        torch.cuda.synchronize(device)
        end_time = time.time()

    except KeyboardInterrupt:
        torch.cuda.synchronize(device)
        end_time = time.time()
        print("\nInterrupted.")

    # 3. Calculate Performance
    total_time = end_time - start_time
    total_ops = iterations * ops_per_matmul
    tflops = (total_ops / total_time) / 1e12

    print("-" * 40)
    print(f"Test Results (GPU {device_id}):")
    print(f"Total Iterations: {iterations}")
    print(f"Total Time:       {total_time:.2f} seconds")
    print(f"Performance:      {tflops:.2f} TFLOPS FP32 MatMul")
    print("-" * 40)
    print(f"Versions: Python {sys.version} Torch: {torch.__version__}")

    del dummy_memory, matrix_a, matrix_b
    torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="GPU ID")
    parser.add_argument("--time", type=int, default=10, help="Duration in seconds")
    parser.add_argument("--with-profiler", action="store_true", help="Run with profiler (optional)")
    args = parser.parse_args()
    p = None
    if args.with_profiler:
        import slim_profiler.profiler as profiler
        gc = profiler.GlobalConstants()
        p = profiler.SlimProfiler(gc, os.getpid(), "profiler", 0.1)
        p.start()

    gpu_stress_test(args.device, args.time)
    if args.with_profiler:
        p.terminate()
