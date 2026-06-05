import logging
import os
import time
import argparse
import sys

import cupy as cp


def gpu_stress_test(device_id: int, duration: int):
    try:
        num_devices = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        raise ValueError("Error: CUDA not detected.")

    if num_devices == 0:
        raise ValueError("Error: CUDA not detected.")

    if device_id >= num_devices:
        raise ValueError(f"Error: GPU index {device_id} not found.")

    with cp.cuda.Device(device_id):
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()
        print(f"--- Target: {name} (ID: {device_id}) ---")

        # 1. Allocation (~5GB)
        # 1.25 billion float32 elements * 4 bytes = 5GB
        print("Allocating ~5GB VRAM...")
        dummy_memory = cp.zeros(1_250_000_000, dtype=cp.float32)

        # 2. Setup Calculation (4096 x 4096 matrix)
        N = 4096
        matrix_a = cp.random.randn(N, N).astype(cp.float32)
        matrix_b = cp.random.randn(N, N).astype(cp.float32)

        ops_per_matmul = 2 * (N**3)

        print(f"Starting stress test for {duration}s...")

        cp.cuda.Device(device_id).synchronize()
        start_time = time.time()
        iterations = 0

        try:
            while time.time() - start_time < duration:
                cp.matmul(matrix_a, matrix_b)
                iterations += 1

            cp.cuda.Device(device_id).synchronize()
            end_time = time.time()

        except KeyboardInterrupt:
            cp.cuda.Device(device_id).synchronize()
            end_time = time.time()
            print("\nInterrupted.")

        total_time = end_time - start_time
        total_ops = iterations * ops_per_matmul
        tflops = (total_ops / total_time) / 1e12

        print("-" * 40)
        print(f"Test Results (GPU {device_id}):")
        print(f"Total Iterations: {iterations}")
        print(f"Total Time:       {total_time:.2f} seconds")
        print(f"Performance:      {tflops:.2f} TFLOPS FP32 MatMul")
        print("-" * 40)
        print(f"Versions: Python {sys.version} CuPy: {cp.__version__}")

        del dummy_memory, matrix_a, matrix_b
        cp.get_default_memory_pool().free_all_blocks()


def main():
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
    if p is not None:
        p.terminate()


if __name__ == "__main__":
    main()
