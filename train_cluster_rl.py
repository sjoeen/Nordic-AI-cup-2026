#!/usr/bin/env python3
"""Driver for the cluster-rl-training package: unpack, set up, train on every training seed,
then evaluate every round's checkpoint against the baseline on the validation seeds.

The package is extracted from cluster-rl-training.zip (never from a git checkout) so its
hash-checked sources keep their exact bytes. All work happens in --workdir, outside the repo.
Rerunning the same command resumes: finished games, merges and evaluations are reused.
Linux only (the package uses POSIX locks and process groups).

--learner neural overlays neural-rl/ onto a separate copy of the package: same games, baseline
and rewards, but the Q table is an MLP ensemble trained on the GPU after every round.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
PACKAGE = "cluster-rl-training"

GIB = 2 ** 30
GAME_GIB = 2.0  # package README's starting allocation per concurrent game


def _read(path):
    try:
        return Path(path).read_text().split()
    except OSError:
        return None


def detect_resources():
    """CPUs and free RAM this container may really use. os.cpu_count() reports the host, not the
    JupyterHub cgroup limit, and oversubscribing it gets workers throttled or OOM-killed."""
    cpus = float(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)
    quota = _read("/sys/fs/cgroup/cpu.max")  # cgroup v2: "<quota|max> <period>"
    if quota and quota[0] != "max":
        cpus = min(cpus, int(quota[0]) / int(quota[1]))
    else:
        v1q, v1p = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if v1q and v1p and int(v1q[0]) > 0:
            cpus = min(cpus, int(v1q[0]) / int(v1p[0]))
    free = None
    meminfo = _read("/proc/meminfo")
    if meminfo and "MemAvailable:" in meminfo:
        free = int(meminfo[meminfo.index("MemAvailable:") + 1]) * 1024
    for limit, used in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        lim, use = _read(limit), _read(used)
        if lim and use and lim[0] != "max" and int(lim[0]) < 2 ** 60:
            room = int(lim[0]) - int(use[0])
            free = room if free is None else min(free, room)
    return cpus, free


def auto_workers():
    cpus, free = detect_resources()
    by_cpu = max(1, int(cpus))
    by_ram = by_cpu if free is None else max(1, int((free / GIB - 2) / GAME_GIB))  # keep 2 GiB headroom
    workers = min(by_cpu, by_ram)
    print(f"resources: {cpus:g} usable CPUs, {'unknown' if free is None else f'{free / GIB:.1f} GiB'} free RAM "
          f"-> {workers} concurrent games (cpu limit {by_cpu}, ram limit {by_ram} at {GAME_GIB:g} GiB/game)", flush=True)
    return workers


def stream(cmd, cwd):
    """Run a command, echo its output live, return the captured stdout lines."""
    print("running:", " ".join(map(str, cmd)), flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
    proc = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line.rstrip("\n"))
    if proc.wait() != 0:
        raise RuntimeError(f"{cmd[2] if len(cmd) > 2 else cmd[0]} failed with exit code {proc.returncode}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default=str(HERE / f"{PACKAGE}.zip"))
    parser.add_argument("--learner", choices=["tabular", "neural"], default="tabular")
    parser.add_argument("--workdir", default=None, help="package is extracted to <workdir>/cluster-rl-training "
                        "(default: ~ for tabular, ~/neural-rl for neural)")
    parser.add_argument("--run", default="runs/all-seeds", help="run directory, relative to the package")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="neural trainer device")
    parser.add_argument("--init-args", nargs=argparse.REMAINDER, default=[],
                        help="extra `init` options, e.g. --init-args --hidden 32 --steps 20000 (must come last)")
    parser.add_argument("--train-seeds", default="1000:1256", help="package default: all 256 training seeds")
    parser.add_argument("--validation-seeds", default="2000:2016")
    parser.add_argument("--test-seeds", default="3000:3032")
    parser.add_argument("--rounds", type=int, default=1, help="total rounds, including those already completed")
    parser.add_argument("--workers", type=int, default=0, help="concurrent games; 0 = size from the container's CPU and RAM limits")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        args.workers = auto_workers()
    neural = args.learner == "neural"
    if args.workdir is None:
        args.workdir = Path.home() / "neural-rl" if neural else Path.home()
    root = Path(args.workdir).resolve() / PACKAGE
    if not (root / "cluster.py").exists():
        print(f"extracting {args.zip} -> {root.parent}", flush=True)
        with zipfile.ZipFile(args.zip) as archive:
            archive.extractall(root.parent)
    py, cluster = sys.executable, root / "cluster.py"
    setup = cluster
    if neural:
        # The overlay is fingerprinted: changing it after a run was initialized requires a new --run.
        (root / "neural").mkdir(exist_ok=True)
        for source in sorted((HERE / "neural-rl").glob("*.py")):
            shutil.copyfile(source, root / "neural" / source.name)
        cluster = root / "neural" / "neural_cluster.py"
        stream([py, "-c", "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', "
                "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"], root)
    t0 = time.perf_counter()

    stream([py, "-m", "pip", "install", "-q", "-r", "requirements.txt"], root)
    stream([py, setup, "setup"], root)
    if not args.skip_tests:
        stream([py, "-m", "unittest", "discover", "-s", "tests"], root)

    run = root / args.run
    if (run / "config.json").exists():
        print(f"{run} already initialized - resuming with its saved seeds/config", flush=True)
    else:
        stream([py, cluster, "init", "--run", run, "--train-seeds", args.train_seeds,
                "--validation-seeds", args.validation_seeds, "--test-seeds", args.test_seeds, *args.init_args], root)

    base_plan = None
    if not args.skip_eval:
        base_plan = stream([py, cluster, "plan-eval", "--run", run, "--split", "validation", "--baseline"], root)[-1]
        stream([py, cluster, "run", "--plan", base_plan, "--workers", args.workers], root)

    # One round at a time so every checkpoint gets a frozen validation score as soon as it exists.
    summaries = {}
    for number in range(args.rounds):
        stream([py, cluster, "train", "--run", run, "--rounds", number + 1, "--workers", args.workers,
                *(["--device", args.device] if neural else [])], root)
        checkpoint = run / "checkpoints" / f"round_{number:04d}.json"
        print(f"round {number} merged at {(time.perf_counter() - t0) / 60:.1f} min -> {checkpoint}", flush=True)
        if args.skip_eval:
            continue
        model_plan = stream([py, cluster, "plan-eval", "--run", run, "--split", "validation",
                             "--checkpoint", checkpoint], root)[-1]
        stream([py, cluster, "run", "--plan", model_plan, "--workers", args.workers], root)
        summaries[checkpoint.name] = json.loads("\n".join(stream(
            [py, cluster, "summary", "--plan", model_plan, "--baseline-plan", base_plan], root)))
        (run / "validation_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print(f"summary saved to {run / 'validation_summary.json'}", flush=True)

    print(f"finished in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
