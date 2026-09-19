#!/usr/bin/env python3
"""Driver for the cluster-rl-training package: unpack, set up, train on every training seed,
then evaluate every round's checkpoint against the baseline on the validation seeds.

The package is extracted from cluster-rl-training.zip (never from a git checkout) so its
hash-checked sources keep their exact bytes. All work happens in --workdir, outside the repo.
Rerunning the same command resumes: finished games, merges and evaluations are reused.
Linux only (the package uses POSIX locks and process groups).
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
PACKAGE = "cluster-rl-training"


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
    parser.add_argument("--workdir", default=str(Path.home()), help="package is extracted to <workdir>/cluster-rl-training")
    parser.add_argument("--run", default="runs/all-seeds", help="run directory, relative to the package")
    parser.add_argument("--train-seeds", default="1000:1256", help="package default: all 256 training seeds")
    parser.add_argument("--validation-seeds", default="2000:2016")
    parser.add_argument("--test-seeds", default="3000:3032")
    parser.add_argument("--rounds", type=int, default=1, help="total rounds, including those already completed")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    root = Path(args.workdir).resolve() / PACKAGE
    if not (root / "cluster.py").exists():
        print(f"extracting {args.zip} -> {root.parent}", flush=True)
        with zipfile.ZipFile(args.zip) as archive:
            archive.extractall(root.parent)
    py, cluster = sys.executable, root / "cluster.py"
    t0 = time.perf_counter()

    stream([py, "-m", "pip", "install", "-q", "-r", "requirements.txt"], root)
    stream([py, cluster, "setup"], root)
    if not args.skip_tests:
        stream([py, "-m", "unittest", "discover", "-s", "tests"], root)

    run = root / args.run
    if (run / "config.json").exists():
        print(f"{run} already initialized - resuming with its saved seeds/config", flush=True)
    else:
        stream([py, cluster, "init", "--run", run, "--train-seeds", args.train_seeds,
                "--validation-seeds", args.validation_seeds, "--test-seeds", args.test_seeds], root)

    base_plan = None
    if not args.skip_eval:
        base_plan = stream([py, cluster, "plan-eval", "--run", run, "--split", "validation", "--baseline"], root)[-1]
        stream([py, cluster, "run", "--plan", base_plan, "--workers", args.workers], root)

    # One round at a time so every checkpoint gets a frozen validation score as soon as it exists.
    summaries = {}
    for number in range(args.rounds):
        stream([py, cluster, "train", "--run", run, "--rounds", number + 1, "--workers", args.workers], root)
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
