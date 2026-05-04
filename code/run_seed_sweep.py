"""Run main.py multiple times with different seeds and collect per-seed
results into ../results/seeds/ for LPIPS noisiness analysis.

Strategy:
  - Run seeds sequentially as subprocesses (clean GPU state between runs)
  - Stop accepting new runs once elapsed + (mean previous run + 10% margin)
    would exceed --time-budget-seconds
  - Always copy {exp}_results.json + {exp}_eval_log.txt into results/seeds/
  - Append a one-line summary per run to results/seeds/sweep_log.txt

Run from the project root (so dataset relative paths in config.py resolve):
  python code/run_seed_sweep.py --seeds 101 102 103 104 105 \
      --time-budget-seconds 23400
"""

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import time


def fmt_hms(seconds: float) -> str:
    return str(dt.timedelta(seconds=int(seconds)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True,
                    help="Seeds to run, in order. Stops early if budget exceeded.")
    ap.add_argument("--time-budget-seconds", type=int, default=6 * 3600 + 25 * 60,
                    help="Total wall-clock budget; no new run is started once "
                         "the next predicted run would exceed it.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Optional override forwarded to main.py.")
    ap.add_argument("--exp-prefix", type=str, default="seed_sweep",
                    help="Per-run experiment_name = '<prefix>_<seed>'.")
    ap.add_argument("--data-split-seed", type=int, default=42,
                    help="Seed for the train/val/test split (held constant "
                         "across runs so all seeds evaluate on the same test "
                         "set). Defaults to 42 to match prior runs.")
    ap.add_argument("--shiny-lpips-max-n", type=int, default=120,
                    help="Cap on shiny-subset render count per run. Lower = "
                         "more seeds fit in budget. 120 keeps each run ~1h.")
    ap.add_argument("--results-dir", type=str,
                    default=os.path.join("results", "seeds"))
    ap.add_argument("--models-dir", type=str, default="./models",
                    help="Where main.py writes <exp>_results.json + eval log.")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    sweep_log = os.path.join(args.results_dir, "sweep_log.txt")

    started_at = time.time()
    durations = []
    completed = []
    skipped = []

    def log(msg):
        line = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(sweep_log, "a") as f:
            f.write(line + "\n")

    log(f"sweep start: seeds={args.seeds} budget={fmt_hms(args.time_budget_seconds)}")

    for seed in args.seeds:
        elapsed = time.time() - started_at
        # Predict next run length: mean of completed runs (with +10% margin),
        # or 0 if we have none yet (always attempt the first run).
        if durations:
            predicted = (sum(durations) / len(durations)) * 1.10
        else:
            predicted = 0
        if elapsed + predicted > args.time_budget_seconds:
            log(f"SKIP seed={seed}: elapsed={fmt_hms(elapsed)} + "
                f"predicted={fmt_hms(predicted)} would exceed budget "
                f"{fmt_hms(args.time_budget_seconds)}")
            skipped.append(seed)
            continue

        exp = f"{args.exp_prefix}_{seed}"
        log(f"START seed={seed} exp={exp}")
        cmd = [
            sys.executable, os.path.join("code", "main.py"),
            "--seed", str(seed),
            "--data-split-seed", str(args.data_split_seed),
            "--experiment-name", exp,
            "--skip-teapot-insertion",
            "--shiny-lpips-max-n", str(args.shiny_lpips_max_n),
        ]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]

        run_log_path = os.path.join(args.results_dir, f"{exp}_stdout.log")
        t0 = time.time()
        # main.py uses relative paths (./dataset/...) -> run with cwd=project
        # root, but add code/ to PYTHONPATH so its `from config import ...`
        # style imports still resolve.
        env = os.environ.copy()
        code_dir = os.path.abspath(os.path.dirname(__file__))
        env["PYTHONPATH"] = code_dir + os.pathsep + env.get("PYTHONPATH", "")
        project_root = os.path.abspath(os.path.join(code_dir, ".."))
        with open(run_log_path, "w") as run_log:
            run_log.write(f"# cmd: {' '.join(cmd)}\n")
            run_log.write(f"# cwd: {project_root}\n")
            run_log.flush()
            proc = subprocess.run(cmd, stdout=run_log,
                                  stderr=subprocess.STDOUT,
                                  cwd=project_root, env=env)
        dt_run = time.time() - t0
        durations.append(dt_run)

        # Copy artifacts. main.py writes:
        #   <models-dir>/<exp>_results.json
        #   <models-dir>/<exp>_eval_log.txt
        for fname in (f"{exp}_results.json", f"{exp}_eval_log.txt"):
            src = os.path.join(args.models_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(args.results_dir, fname))
            else:
                log(f"  WARN missing artifact: {src}")

        if proc.returncode == 0:
            completed.append(seed)
            log(f"DONE  seed={seed} duration={fmt_hms(dt_run)} "
                f"rc={proc.returncode}")
        else:
            log(f"FAIL  seed={seed} duration={fmt_hms(dt_run)} "
                f"rc={proc.returncode} (see {run_log_path})")

    total = time.time() - started_at
    log(f"sweep end: completed={completed} skipped={skipped} "
        f"total={fmt_hms(total)}")


if __name__ == "__main__":
    main()
