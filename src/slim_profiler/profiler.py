import argparse
import dataclasses
import itertools
import json
import logging
import os
import signal
import sys
import threading
import time
from collections import defaultdict

import psutil

import slim_profiler

try:
    import pynvml
except ImportError:
    pynvml = None

from typing import Mapping, Any, List, Dict, Tuple, Set

PSUTIL_NOTFOUND_ERRORS = (
    psutil.NoSuchProcess,
    psutil.ZombieProcess,
    psutil.AccessDenied,
    psutil.Error,
    OSError,
    IOError,
)
DEV_NULL = "nul" if sys.platform == "win32" else "/dev/null"

_lh = logging.getLogger("SlimProfiler")
__all__ = ["SlimProfiler", "GlobalConstants"]


class GlobalConstants:
    num_gpus: int
    gpu_names = []
    gpu_mems = []
    total_mem: float
    total_gpu_mem: float
    num_cores: int
    nvidia_driver_version: str
    nvidia_cuda_max_supported: str

    def __init__(self):
        self.gpu_names = []
        self.gpu_mems = []
        self.num_cores = psutil.cpu_count(logical=True)
        _lh.info("Num cores: %d", self.num_cores)
        self.total_mem = psutil.virtual_memory().total
        _lh.info("Total memory: %.2f MiB", self.total_mem / (1 << 20))
        if pynvml is None:
            self.num_gpus = 0
            self.total_gpu_mem = 0
            self.nvidia_driver_version = "N/A"
            self.nvidia_cuda_max_supported = ""
        else:
            self.total_gpu_mem = 0
            try:
                pynvml.nvmlInit()
                self.nvidia_driver_version = pynvml.nvmlSystemGetDriverVersion()
                cuda_version_raw = pynvml.nvmlSystemGetCudaDriverVersion()
                cuda_major = cuda_version_raw // 1000
                cuda_minor = (cuda_version_raw % 1000) // 10
                self.nvidia_cuda_max_supported = f"{cuda_major}.{cuda_minor}"
                _lh.info("NVML Driver Version: %s supporting CUDA: %s", self.nvidia_driver_version, self.nvidia_cuda_max_supported)

                self.num_gpus = pynvml.nvmlDeviceGetCount()
                for i in range(self.num_gpus):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    this_gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total
                    this_gpu_name = pynvml.nvmlDeviceGetName(handle)
                    self.gpu_names.append(this_gpu_name)
                    self.gpu_mems.append(this_gpu_mem)
                    _lh.info("Found GPU %d: %s MEM %.2f MiB", i, pynvml.nvmlDeviceGetName(handle), this_gpu_mem)
                    self.total_gpu_mem += this_gpu_mem

                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                self.num_gpus = 0
        _lh.info("Found %d GPUs", self.num_gpus)
        _lh.info("Total GPU memory: %.2f MiB", self.total_gpu_mem / (1 << 20))


def get_gpu_vmem_utilization(
    global_constants: GlobalConstants, pids: Set[int]
) -> List[Tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if pynvml is None:
        return []
    retl = []
    # From nvitop:
    # Only utilization samples that were recorded after this timestamp will be returned.
    # The CPU timestamp, i.e. absolute Unix epoch timestamp (in microseconds), is used.
    # Here we use the timestamp 1 second ago to ensure the record buffer is not empty.
    try:
        pynvml.nvmlInit()
        for i in range(global_constants.num_gpus):
            vmem = defaultdict(lambda: 0)
            utilization = defaultdict(lambda: 0)
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            except pynvml.NVMLError as e:
                _lh.error("get_gpu_vmem_utilization ERR: %s", e)
                continue
            if  hasattr(pynvml, "nvmlDeviceGetComputeRunningProcesses"):
                try:
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                except pynvml.NVMLError as e:
                    _lh.error("nvmlDeviceGetComputeRunningProcesses ERR: %s", e)
                    processes = []
            else:
                processes = []

            try:
                # last_seen_timestamp=0 gets the most recent sample
                util_samples = pynvml.nvmlDeviceGetProcessUtilization(handle, 0)
                for sample in util_samples:
                    if sample.pid in pids:
                        utilization[str(sample.pid)] = sample.smUtil
            except pynvml.NVMLError as e:
                if str(e) == "Not Found":
                    pass # No idea why this emerge
                else:
                    _lh.error("nvmlDeviceGetProcessUtilization ERR: %s", e)

            for process in processes:
                if process.pid in pids:
                    if process.usedGpuMemory is None:
                        vmem[str(process.pid)] = 0
                    else:
                        vmem[str(process.pid)] = process.usedGpuMemory

            retl.append((vmem, utilization))
        pynvml.nvmlShutdown()
    except pynvml.NVMLError as e:
        retl = [(defaultdict(lambda: 0), defaultdict(lambda: 0))] * global_constants.num_gpus
        _lh.error("get_gpu_vmem_utilization ERR: %s", e)
    return retl


@dataclasses.dataclass
class ProcessLocalData:
    mem_json_d: Dict[str, float]
    cpu_json_d: Dict[str, float]
    scpids: List[str]
    """Process IDs as strings"""

    gpu_vmem_d: List[Mapping[str, Any]]
    gpu_util_d: List[Mapping[str, Any]]

    cpu_time_cache: Dict[int, Tuple[int, float]]
    """ # Process -> [Last time point, Total CPU Time]"""
    max_rss_cache: float
    wallclock_start: int
    max_gpu_mem_cache: float

    def __init__(self):
        self.mem_json_d = {}
        self.cpu_json_d = {}
        self.scpids = []
        self.gpu_vmem_d = []
        self.gpu_util_d = []
        self.cpu_time_cache = {}
        self.max_rss_cache = 0
        self.wallclock_start = time.time_ns()
        self.max_gpu_mem_cache = 0


class Serializer:
    _dst_tsv: str
    _global_constants: GlobalConstants

    def __init__(self, global_constants: GlobalConstants, dst_tsv: str):
        self._dst_tsv = dst_tsv
        self._perproc_plot_appender = open(self._dst_tsv + ".perproc_plot.tsv", "w", encoding="UTF-8")
        self._joint_plot_appender = open(self._dst_tsv + ".joint_plot.tsv", "w", encoding="UTF-8")
        self._global_constants = global_constants
        with open(f"{dst_tsv}.gc.json", "w", encoding="UTF-8") as w:
            json.dump(
                {
                    "gpus":[
                        {
                            "name": self._global_constants.gpu_names[i],
                            "mem": self._global_constants.gpu_mems[i],
                        }
                        for i in range(self._global_constants.num_gpus)
                    ],
                    "cpus": self._global_constants.num_cores,
                    "mem": self._global_constants.total_mem,
                    "software":{
                        "python": sys.version,
                        "psutil": psutil.__version__,
                        "slim_profiler": slim_profiler.__version__,
                        "pynvml": "PRESENT" if pynvml is not None else "N/A",
                        "nvidia_driver": self._global_constants.nvidia_driver_version,
                        "nvidia_cuda_max_supported": self._global_constants.nvidia_cuda_max_supported,
                    }
                },
                w,
                indent=4,
            )
        perproc_table_header = ["TIME", "PID", "MEM_RSS", "CPU_UTIL_PCT"]
        for i in range(global_constants.num_gpus):
            perproc_table_header.append(f"GPU{i}_VMEM")
            perproc_table_header.append(f"GPU{i}_UTIL_PCT")

        joint_table_header = ["TIME", "MEM_RSS", "CPU_UTIL_PCT"]
        for i in range(global_constants.num_gpus):
            joint_table_header.append(f"GPU{i}_VMEM")
            joint_table_header.append(f"GPU{i}_UTIL_PCT")

        self._perproc_plot_appender.write("\t".join(perproc_table_header))
        self._perproc_plot_appender.write("\n")

        self._joint_plot_appender.write("\t".join(joint_table_header))
        self._joint_plot_appender.write("\n")

    def close(self):
        self._perproc_plot_appender.close()
        self._joint_plot_appender.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def serialize(self, timestamp: float, pld: ProcessLocalData):
        joint_cpu = 0.0
        joint_mem = 0
        for k in pld.scpids:
            joint_cpu += pld.cpu_json_d[k]
            joint_mem += pld.mem_json_d[k]
            self._perproc_plot_appender.write(
                "\t".join(
                    (
                        str(timestamp),
                        str(k),
                        str(pld.mem_json_d[k]),
                        str(pld.cpu_json_d[k]),
                        *itertools.chain(
                            *zip(
                                [str(pld.gpu_vmem_d[i][str(k)]) for i in range(self._global_constants.num_gpus)],
                                [str(pld.gpu_util_d[i][str(k)]) for i in range(self._global_constants.num_gpus)],
                            ),
                        ),
                    )
                )
            )
            self._perproc_plot_appender.write("\n")
            self._perproc_plot_appender.flush()

        self._joint_plot_appender.write(
            "\t".join(
                (
                    str(timestamp),
                    str(joint_mem),
                    str(joint_cpu),
                        *itertools.chain(
                            *zip(
                                [str(sum(pld.gpu_vmem_d[i].values())) for i in range(self._global_constants.num_gpus)],
                                [str(sum(pld.gpu_util_d[i].values())) for i in range(self._global_constants.num_gpus)],
                            ),
                        ),
                )
            )
        )
        joint_gpu_mem = sum(sum(pld.gpu_vmem_d[i].values()) for i in range(self._global_constants.num_gpus))
        self._joint_plot_appender.write("\n")
        self._joint_plot_appender.flush()
        if joint_mem > pld.max_rss_cache:
            pld.max_rss_cache = joint_mem
        if joint_gpu_mem > pld.max_gpu_mem_cache:
            pld.max_gpu_mem_cache = joint_gpu_mem


class SlimProfiler(threading.Thread):
    _should_stop: bool
    _trace_pid: int
    _dst_tsv: str
    _p: psutil.Process
    _interval: float
    _pld: ProcessLocalData
    _global_constants: GlobalConstants

    def __init__(self, global_constants: GlobalConstants, trace_pid: int, dst_tsv: str, interval: float) -> None:
        super(SlimProfiler, self).__init__()
        self._global_constants = global_constants
        self._pld = ProcessLocalData()
        self._should_stop = False
        self._trace_pid = trace_pid
        try:
            self._p = psutil.Process(self._trace_pid)
        except PSUTIL_NOTFOUND_ERRORS:
            self.terminate()
        self._dst_tsv = dst_tsv
        self._interval = interval
        self._max_rss_cache = 0

    def _init(self):
        self._pld.wallclock_start = time.time_ns()
        self._pld.cpu_time_cache = {}
        try:
            children = list(self._p.children(recursive=True))
            children.append(self._p)
        except PSUTIL_NOTFOUND_ERRORS:
            self.terminate()
            return
        for child in children:
            current_time = time.time_ns()
            current_cpu_times = child.cpu_times()
            self._pld.cpu_time_cache[child.pid] = (current_time, current_cpu_times.user + current_cpu_times.system)

    def _collect(self):
        try:
            children = list(self._p.children(recursive=True))
            children.append(self._p)
        except PSUTIL_NOTFOUND_ERRORS:
            self.terminate()
            children = []
        if not self._p.is_running():
            self.terminate()
        _lh.debug("Collecting data for %d processes", len(children))
        vu = get_gpu_vmem_utilization(
            self._global_constants,
            set(c.pid for c in children),
        )
        self._pld.gpu_vmem_d = []
        self._pld.gpu_util_d = []
        for device_gpu_vmem_d, device_gpu_util_d in vu:
            self._pld.gpu_vmem_d.append(device_gpu_vmem_d)
            self._pld.gpu_util_d.append(device_gpu_util_d)
        self._pld.scpids = []
        for child in children:
            try:
                cpid = child.pid
                scpid = str(cpid)
                current_cpu_times = child.cpu_times()
                current_time = time.time_ns()
                current_cpu_time = current_cpu_times.user + current_cpu_times.system
                last_cpu_time_cache = self._pld.cpu_time_cache.get(child.pid, (current_time, 0))
                self._pld.cpu_time_cache[child.pid] = (current_time, current_cpu_time)
                try:
                    cpu_pct = (
                        (current_cpu_time - last_cpu_time_cache[1])
                        / (current_time - last_cpu_time_cache[0])
                        * 1e9
                        * 100
                    )
                    self._pld.cpu_json_d[scpid] = cpu_pct
                except ZeroDivisionError:
                    self._pld.cpu_json_d[scpid] = -1
                current_rss = child.memory_info().rss
                self._pld.mem_json_d[scpid] = current_rss
                self._pld.scpids.append(scpid)
            except PSUTIL_NOTFOUND_ERRORS as e:
                _lh.error("Process %d not found: %s", child.pid, e)
                continue
            except RuntimeError:
                self.terminate()
                break

    def run(self):
        _lh.info("started with PID=%d", self._trace_pid)
        serializer = Serializer(self._global_constants, self._dst_tsv)
        self._init()
        while not self._should_stop:
            timestamp = time.time()
            self._collect()
            serializer.serialize(timestamp, self._pld)
            time.sleep(self._interval)

        _lh.info(
            "Process group peak RSS: %d MiB (%.2f%%)",
            self._pld.max_rss_cache / (1 << 20),
            self._pld.max_rss_cache / self._global_constants.total_mem * 100,
        )
        if  self._global_constants.total_gpu_mem > 0:
            _lh.info(
                "Process group peak GPU Mem: %d MiB (%.2f%%)",
                self._pld.max_gpu_mem_cache / (1 << 20),
                self._pld.max_gpu_mem_cache / self._global_constants.total_gpu_mem * 100,
            )
        _lh.info(
            "Process group mean CPU Utilization: %.2f%%",
            sum(x[1] for x in self._pld.cpu_time_cache.values())
            / (time.time_ns() - self._pld.wallclock_start)
            * 1e9
            * 100,
        )
        _lh.info("Finished")

    def terminate(self):
        _lh.info("SIGTERM received")
        self._should_stop = True


def main():
    # Initialize the logger, setting filter to INFO
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="SlimProfiler")
    parser.add_argument("--trace-pid", type=int, required=False, help="PID to trace", default=os.getppid())
    parser.add_argument("--report_global_constants", action="store_true", help="Report global constants and exit")
    parser.add_argument("--dst-tsv", type=str, required=False, help="Prefix of the destination TSV files")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval in seconds")
    args = parser.parse_args()
    gc = GlobalConstants()
    if args.report_global_constants:
        sys.exit(0)
    if args.trace_pid is None:
        _lh.error("Specify the PID!")
        sys.exit(1)
    if args.dst_tsv is None:
        _lh.error("Specify the prefix to destination TSV!")
        sys.exit(1)

    sp = SlimProfiler(gc, args.trace_pid, args.dst_tsv, args.interval)
    # Register SIGTERM handler to sp.terminate()
    signal.signal(signal.SIGTERM, lambda _: sp.terminate())
    signal.signal(signal.SIGINT, lambda _: sp.terminate())
    sp.run() # Use start in other scenarios.

if __name__ == "__main__":
    main()
