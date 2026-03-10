import json
import logging
import argparse

from matplotlib.ticker import FuncFormatter

import slim_profiler

import pandas as pd
import matplotlib.pyplot as plt

_lh = logging.getLogger(__name__)

def format_si(x, pos):
    """
    x is the value, pos is the tick position
    """
    for unit in ['', 'k', 'M', 'G', 'T']:
        if abs(x) < 1024.0:
            return f"{x:g}{unit}"
        x /= 1024.0
    return f"{x:g}P"

def plot_main(gc_json:str, joint_plot_tsv: str, plot_prefix: str) -> None:
    with open(gc_json, 'r', encoding="UTF-8") as f:
        gc_data = json.load(f)
    if gc_data["software"]["slim_profiler"] != slim_profiler.__version__:
        _lh.warning("The data was collected with slim_profiler version %s, but you are using version %s. This may lead to compatibility issues.", gc_data["software"]["slim_profiler"], slim_profiler.__version__)
    df = pd.read_csv(joint_plot_tsv, sep="\t")
    df["TIME"] = pd.to_datetime(df["TIME"], unit="s")
    # Plot memory. Only use MEM_RSS
    plt.figure(figsize=(12, 6))
    plt.plot(df["TIME"], df["MEM_RSS"], label="RSS")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Memory")
    plt.title("Memory Usage Over Time")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_si))

    mean_rss = df["MEM_RSS"].mean()
    if mean_rss > 0.5 * gc_data["mem"]:
        plt.axhline(gc_data["mem"], color="red", linestyle="--", label="Total Memory")
        plt.ylim(0, gc_data["mem"] * 1.2)
    else:
        plt.ylim(0, df["MEM_RSS"].max() * 1.2)

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}.memory_usage.png")
    plt.clf()

    # Now plot CPU.
    plt.figure(figsize=(12, 6))
    plt.plot(df["TIME"], df["CPU_UTIL_PCT"], label="CPU Usage (%)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    mean_cpu = df["CPU_UTIL_PCT"].mean()
    if mean_cpu > 0.5 * gc_data["cpus"] * 100:
        plt.axhline(gc_data["cpus"] * 100, color="red", linestyle="--", label="Total CPU")
        plt.ylim(0, gc_data["cpus"] * 100)
    else:
        plt.ylim(0, df["CPU_UTIL_PCT"].max() * 1.2)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}.cpu_usage.png")
    plt.clf()

    # Plotting GPUs.
    num_gpus = len(gc_data["gpus"])

    # Plot all GPU memory usage in one plot.
    plt.figure(figsize=(12, 6))
    fig, axs = plt.subplots(num_gpus, 1, figsize=(12, 6 * num_gpus), sharex=True)
    if num_gpus == 1:
        axs = [axs]
    for i in range(num_gpus):
        gpu_id = i
        gpu_mem_col = f"GPU{gpu_id}_VMEM"
        axs[i].plot(df["TIME"], df[gpu_mem_col], label=f"GPU {gpu_id} Memory Used")
        mean_gpu_mem = df[gpu_mem_col].mean()
        if mean_gpu_mem > 0.5 * gc_data["gpus"][i]["mem"]:
                axs[i].axhline(gc_data["gpus"][i]["mem"], color="red", linestyle="--", label="Total GPU Memory")
                axs[i].set_ylim(0, gc_data["gpus"][i]["mem"] * 1.2)
        else:
                axs[i].set_ylim(0, df[gpu_mem_col].max() * 1.2)
        axs[i].set_ylabel("Memory")
        axs[i].yaxis.set_major_formatter(FuncFormatter(format_si))
        axs[i].legend()
        axs[i].grid()
    axs[-1].set_xlabel("Time (UTC)")
    plt.suptitle("GPU Memory Usage Over Time")
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}.gpu_memory_usage.png")
    plt.clf()

    # Plot all GPU utilization in one plot.
    plt.figure(figsize=(12, 6))
    fig, axs = plt.subplots(num_gpus, 1, figsize=(12, 6 * num_gpus), sharex=True)
    if num_gpus == 1:
        axs = [axs]
    for i in range(num_gpus):
        gpu_id = i
        gpu_util_col = f"GPU{gpu_id}_UTIL_PCT"
        axs[i].plot(df["TIME"], df[gpu_util_col], label=f"GPU {gpu_id} Utilization (%)")
        mean_gpu_util = df[gpu_util_col].mean()
        if mean_gpu_util > 0.5 * 100:
                axs[i].axhline(100, color="red", linestyle="--", label="Total GPU Utilization")
                axs[i].set_ylim(0, 120)
        else:
                axs[i].set_ylim(0, df[gpu_util_col].max() * 1.2)
        axs[i].set_ylabel("GPU Utilization (%)")
        axs[i].legend()
        axs[i].grid()
    axs[-1].set_xlabel("Time (UTC)")
    plt.suptitle("GPU Utilization Over Time")
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}.gpu_utilization.png")
    plt.clf()

def main():
    logging.basicConfig(level="INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst-tsv", type=str, required=True, help="The --dst-tsv you used in data collection")
    args = parser.parse_args()
    plot_main(
        f"{args.dst_tsv}.gc.json",
        f"{args.dst_tsv}.joint_plot.tsv",
        f"{args.dst_tsv}.plot"
    )

if __name__ == "__main__":
    main()
