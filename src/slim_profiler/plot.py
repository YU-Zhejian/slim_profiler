import json
import logging
import argparse

import matplotlib

matplotlib.use('Agg')

from matplotlib.ticker import FuncFormatter

import slim_profiler

import polars as pl
import matplotlib.pyplot as plt

_lh = logging.getLogger("SlimProfiler::Plot")


def format_si(x, pos):
    """
    x is the value, pos is the tick position
    """
    for unit in ["", "k", "M", "G", "T"]:
        if abs(x) < 1024.0:
            return f"{x:g}{unit}"
        x /= 1024.0
    return f"{x:g}P"


def plot_main(
    gc_json: str,
    joint_plot_tsv: str,
    plot_prefix: str,
    base_fig_width: int = 12,
    base_fig_height: int = 6,
    fig_ext: str = "png",
) -> None:
    with open(gc_json, "r", encoding="UTF-8") as f:
        gc_data = json.load(f)
    if gc_data["software"]["slim_profiler"] != slim_profiler.__version__:
        _lh.warning(
            "The data was collected with slim_profiler version %s, but you are using version %s. This may lead to compatibility issues.",
            gc_data["software"]["slim_profiler"],
            slim_profiler.__version__,
        )
    df = pl.read_csv(joint_plot_tsv, separator="\t")
    # TIME is written as fractional epoch seconds; convert through microseconds to keep sub-second precision.
    df = df.with_columns(
        pl.from_epoch((pl.col("TIME") * 1_000_000).round().cast(pl.Int64), time_unit="us").alias("TIME")
    )
    time_axis = df["TIME"].to_numpy()
    # Plot memory. Only use MEM_RSS
    plt.figure(figsize=(base_fig_width, base_fig_height))
    mem_rss = df["MEM_RSS"].to_numpy()
    plt.plot(time_axis, mem_rss, label="RSS")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Memory")
    plt.title("Memory Usage Over Time")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_si))

    mean_rss = mem_rss.mean()
    if mean_rss > 0.5 * gc_data["mem"]:
        plt.axhline(gc_data["mem"], color="red", linestyle="--", label="Total Memory")
        plt.ylim(0, gc_data["mem"] * 1.2)
    else:
        plt.ylim(0, mem_rss.max() * 1.2)

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}.memory_usage.{fig_ext}")
    plt.close()

    # Now plot CPU.
    plt.figure(figsize=(base_fig_width, base_fig_height))
    cpu_util = df["CPU_UTIL_PCT"].to_numpy()
    plt.plot(time_axis, cpu_util, label="CPU Usage (%)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    mean_cpu = cpu_util.mean()
    if mean_cpu > 0.5 * gc_data["cpus"] * 100:
        plt.axhline(gc_data["cpus"] * 100, color="red", linestyle="--", label="Total CPU")
        plt.ylim(0, gc_data["cpus"] * 100)
    else:
        plt.ylim(0, cpu_util.max() * 1.2)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}.cpu_usage.{fig_ext}")
    plt.close()

    # Plotting GPUs.
    num_gpus = len(gc_data["gpus"])
    gpu_id_in_use = []
    for i in range(num_gpus):
        gpu_mem_col = f"GPU{i}_VMEM"
        gpu_util_col = f"GPU{i}_UTIL_PCT"
        if df[gpu_mem_col].to_numpy().sum() > 0 or df[gpu_util_col].to_numpy().sum() > 0:
            gpu_id_in_use.append(i)
    if not gpu_id_in_use:
        _lh.info("No GPU usage detected, skipping GPU plots.")
        return

    # Plot all GPU memory usage in one plot.
    fig, axs = plt.subplots(len(gpu_id_in_use), 1, figsize=(base_fig_width, base_fig_height * len(gpu_id_in_use)), sharex=True)
    if len(gpu_id_in_use) == 1:
        axs = [axs]
    for i, gpu_id in enumerate(gpu_id_in_use):
        gpu_mem_col = f"GPU{gpu_id}_VMEM"
        gpu_mem = df[gpu_mem_col].to_numpy()
        axs[i].plot(time_axis, gpu_mem, label=f"GPU {gpu_id} Memory Used")
        mean_gpu_mem = gpu_mem.mean()
        total_gpu_mem = gc_data["gpus"][gpu_id]["mem"]
        if mean_gpu_mem > 0.5 * total_gpu_mem:
            axs[i].axhline(total_gpu_mem, color="red", linestyle="--", label="Total GPU Memory")
            axs[i].set_ylim(0, total_gpu_mem * 1.2)
        else:
            axs[i].set_ylim(0, gpu_mem.max() * 1.2)
        axs[i].set_ylabel("Memory")
        axs[i].yaxis.set_major_formatter(FuncFormatter(format_si))
        axs[i].legend()
        axs[i].grid()
    axs[-1].set_xlabel("Time (UTC)")
    plt.suptitle("GPU Memory Usage Over Time")
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}.gpu_memory_usage.{fig_ext}")
    plt.close()

    # Plot all GPU utilization in one plot.
    fig, axs = plt.subplots(len(gpu_id_in_use), 1, figsize=(base_fig_width, base_fig_height * len(gpu_id_in_use)), sharex=True)
    if len(gpu_id_in_use) == 1:
        axs = [axs]
    for i, gpu_id in enumerate(gpu_id_in_use):
        gpu_util_col = f"GPU{gpu_id}_UTIL_PCT"
        gpu_util = df[gpu_util_col].to_numpy()
        axs[i].plot(time_axis, gpu_util, label=f"GPU {gpu_id} Utilization (%)")
        mean_gpu_util = gpu_util.mean()
        if mean_gpu_util > 0.5 * 100:
            axs[i].axhline(100, color="red", linestyle="--", label="Total GPU Utilization")
            axs[i].set_ylim(0, 120)
        else:
            axs[i].set_ylim(0, gpu_util.max() * 1.2)
        axs[i].set_ylabel("GPU Utilization (%)")
        axs[i].legend()
        axs[i].grid()
    axs[-1].set_xlabel("Time (UTC)")
    plt.suptitle("GPU Utilization Over Time")
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}.gpu_utilization.{fig_ext}")
    plt.close()


def main():
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    _lh.info("Version %s. Using matplotlib %s", slim_profiler.__version__, matplotlib.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst-tsv", type=str, required=True, help="The --dst-tsv you used in data collection")
    args = parser.parse_args()
    plot_main(f"{args.dst_tsv}.gc.json", f"{args.dst_tsv}.joint_plot.tsv", f"{args.dst_tsv}.plot")


if __name__ == "__main__":
    main()
