"""Reproduce the supplied MP1 baseline and measure CPU evaluation RAM.

Run with the MP1 Python environment. The script verifies dependencies, trains
the original GPT in FP32, then scores validation and test on CPU in FP32.
Pass --checkpoint to measure and score an existing model without retraining.
Use --split validation while developing; score test only after freezing.
"""

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "MP1_student_starter" / "code"


def check_environment(device: str) -> None:
    import numpy
    import tokenizers
    import torch

    expected = {
        "Python": (sys.version_info.major, sys.version_info.minor) == (3, 12),
        "PyTorch": torch.__version__.split("+")[0] == "2.7.1",
        "NumPy": numpy.__version__ == "2.5.3",
        "tokenizers": tokenizers.__version__ == "0.21.4",
    }
    failures = [name for name, passed in expected.items() if not passed]
    print(f"Python executable: {sys.executable}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"NumPy: {numpy.__version__}; tokenizers: {tokenizers.__version__}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("CUDA GPU: unavailable", flush=True)
    if device == "cuda" and not torch.cuda.is_available():
        failures.append("CUDA availability (requested --device cuda)")
    if failures:
        names = ", ".join(failures)
        raise SystemExit(
            f"Environment check failed: {names}. Use Python 3.12 and "
            "install the exact dependencies listed in the README."
        )
    print("Environment check passed.\n", flush=True)


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


def process_parent_map(kernel32) -> dict[int, int]:
    create_snapshot = kernel32.CreateToolhelp32Snapshot
    create_snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    create_snapshot.restype = wintypes.HANDLE
    snapshot = create_snapshot(0x00000002, 0)  # all processes
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())

    first = kernel32.Process32FirstW
    first.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    first.restype = wintypes.BOOL
    next_process = kernel32.Process32NextW
    next_process.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    next_process.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    parents = {}
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        present = first(snapshot, ctypes.byref(entry))
        while present:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            present = next_process(snapshot, ctypes.byref(entry))
    finally:
        close_handle(snapshot)
    return parents


def process_memory_bytes(process_handle) -> tuple[int, int]:
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("Psapi.dll")
    get_memory_info = psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]
    get_memory_info.restype = wintypes.BOOL
    if not get_memory_info(process_handle, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return int(counters.PeakWorkingSetSize), int(counters.WorkingSetSize)


def run(command: list[str], *, measure_ram: bool = False) -> dict | None:
    print("\n> " + subprocess.list2cmdline(command), flush=True)
    if not measure_ram:
        subprocess.run(command, cwd=CODE_DIR, check=True)
        return None

    if sys.platform != "win32":
        raise RuntimeError("Peak RAM measurement currently uses the Windows process API.")

    process = subprocess.Popen(command, cwd=CODE_DIR)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    open_process.restype = wintypes.HANDLE
    wait_for_single_object = kernel32.WaitForSingleObject
    wait_for_single_object.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    wait_for_single_object.restype = wintypes.DWORD
    handles = {}
    individual_peak_bytes = {}
    sampled_tree_peak_bytes = 0
    try:
        # The Windows venv python.exe can launch a second python.exe. Measure
        # the whole descendant tree instead of just the small launcher stub.
        runner_pid = os.getpid()
        handles[runner_pid] = open_process(0x00100410, False, runner_pid)
        handles[process.pid] = open_process(0x00100410, False, process.pid)
        if not handles[runner_pid] or not handles[process.pid]:
            raise ctypes.WinError(ctypes.get_last_error())
        known_pids = {runner_pid, process.pid}
        evaluation_pids = {process.pid}
        while True:
            parents = process_parent_map(kernel32)
            for pid, parent_pid in parents.items():
                if parent_pid not in known_pids or pid in known_pids:
                    continue
                handle = open_process(0x00100410, False, pid)
                if handle:
                    handles[pid] = handle
                    known_pids.add(pid)
                    if parent_pid in evaluation_pids:
                        evaluation_pids.add(pid)

            sampled_tree_bytes = 0
            for pid, handle in handles.items():
                peak_bytes, current_bytes = process_memory_bytes(handle)
                individual_peak_bytes[pid] = max(
                    individual_peak_bytes.get(pid, 0), peak_bytes
                )
                if wait_for_single_object(handle, 0) == 0x102:
                    sampled_tree_bytes += current_bytes
            sampled_tree_peak_bytes = max(sampled_tree_peak_bytes, sampled_tree_bytes)

            # Keep sampling after the launcher exits if its evaluation child remains.
            if not any(
                wait_for_single_object(handles[pid], 0) == 0x102
                for pid in evaluation_pids
                if pid in handles
            ):
                break
            time.sleep(0.02)
    finally:
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        for handle in handles.values():
            close_handle(handle)

    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    # Summing each process's high-water mark is conservative: their individual
    # peaks need not occur at the same instant, so this cannot understate the
    # peak for the evaluated process tree.
    peak_ram_upper_bound = sum(individual_peak_bytes.values())
    return {
        "sampled_peak_process_tree_working_set_bytes": sampled_tree_peak_bytes,
        "peak_ram_upper_bound_bytes": peak_ram_upper_bound,
        "peak_ram_upper_bound_gib": peak_ram_upper_bound / (1024**3),
        "ram_limit_gib": 4.0,
        "ram_limit_pass": peak_ram_upper_bound <= 4 * 1024**3,
        "measurement": "Windows process-tree working set; conservative sum of per-process PeakWorkingSetSize",
        "scope": "run_baseline.py process plus evaluate.py process tree; includes imports, data/model loading, scoring, and saving",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--split", choices=("validation", "test", "both"), default="both")
    parser.add_argument("--output-dir", type=Path,
                        help="Write evaluation evidence to a separate directory.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Measure and score an existing checkpoint without retraining.",
    )
    args = parser.parse_args()

    check_environment("cpu" if args.checkpoint is not None else args.device)

    if args.checkpoint is not None and args.run_dir is not None:
        raise SystemExit("Use either --checkpoint or --run-dir, not both.")
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.resolve()
        if not checkpoint.is_file():
            raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
        run_dir = checkpoint.parent
    elif args.run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = ROOT / "experiments" / f"baseline-reproduction-{stamp}" / "run"
    else:
        run_dir = args.run_dir.resolve()
    if args.checkpoint is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(f"Run directory already contains files: {run_dir}")
        common = ["--threads", "4"]
        run(
            [
                sys.executable,
                "train.py",
                "--implementation",
                "model",
                "--device",
                args.device,
                "--precision",
                "fp32",
                *common,
                "--seed",
                "17",
                "--steps",
                "1200",
                "--batch-size",
                "32",
                "--eval-every",
                "300",
                "--run-dir",
                str(run_dir),
            ]
        )
        checkpoint = run_dir / "checkpoint.pt"

    common = ["--threads", "4"]
    output_dir = args.output_dir.resolve() if args.output_dir else checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    resource_measurements = {}
    splits = ("validation", "test") if args.split == "both" else (args.split,)
    for split in splits:
        result_path = output_dir / f"{split}_cpu_fp32.json"
        ram = run(
            [
                sys.executable,
                "evaluate.py",
                "--checkpoint",
                str(checkpoint),
                "--device",
                "cpu",
                "--precision",
                "fp32",
                *common,
                "--split",
                split,
                "--output",
                str(result_path),
            ],
            measure_ram=True,
        )
        resource_measurements[split] = ram
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result.update(ram)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    resource_path = output_dir / "resource_measurements.json"
    resource_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "cpu_fp32_evaluation": resource_measurements,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nCPU FP32 evaluation complete. Results are in: {output_dir}", flush=True)
    print(f"RAM measurements: {resource_path}", flush=True)


if __name__ == "__main__":
    main()
