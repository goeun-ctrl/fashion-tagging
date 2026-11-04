"""Run a list of train_eval.py jobs on several GPUs, one job per GPU at a time.

    python common/tools/run_queue.py --jobs experiments/role5_jobs.txt --gpus 0 1 2 3 \
        --train_dir data/train --val_dir data/val --out_dir runs --results_csv results/results.csv
    python common/tools/run_queue.py --jobs ... --only P1          # only the lines tagged [P1]
    python common/tools/run_queue.py --jobs ... --dry_run          # print the commands, run nothing

Job file: one job per line = train_eval.py arguments (must contain --exp_id). '#' starts a comment.
A leading tag like "[P1]" groups jobs for --only. A job is skipped when runs/<exp_id>/preds_val.npz
already exists (finished), so the queue can be restarted after an interruption; an unfinished run
restarts from scratch (train_eval.py has no resume). Each job's console output goes to
<out_dir>/<exp_id>.log. Run the queue itself under nohup or tmux so it survives a dropped SSH session.
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import time

COMMON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # common/


def parse_jobs(path, only):
    jobs = []
    for raw in open(path, encoding="utf-8"):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"\[(\w+)\]\s*(.*)", line)
        tag, line = (m.group(1), m.group(2)) if m else ("", line)
        if only and tag not in only:
            continue
        args = shlex.split(line)
        if "--exp_id" not in args:
            raise ValueError(f"job without --exp_id: {raw.strip()}")
        jobs.append((args[args.index("--exp_id") + 1], args))
    ids = [j for j, _ in jobs]
    dup = {j for j in ids if ids.count(j) > 1}
    if dup:
        raise ValueError(f"duplicate exp_id in job file: {sorted(dup)}")
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpus", nargs="+", default=["0"])
    ap.add_argument("--only", nargs="*", default=[], help="run only these tags, e.g. P1 P2")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--results_csv", default=os.path.join("results", "results.csv"))
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    todo = []
    for exp_id, args in parse_jobs(a.jobs, a.only):
        if os.path.exists(os.path.join(a.out_dir, exp_id, "preds_val.npz")):
            print(f"skip {exp_id} (already finished)")
            continue
        cmd = [sys.executable, os.path.join(COMMON, "train_eval.py"), *args, "--train_dir", a.train_dir,
               "--val_dir", a.val_dir, "--out_dir", a.out_dir, "--results_csv", a.results_csv]
        todo.append((exp_id, cmd))
    print(f"{len(todo)} job(s) on GPU(s) {', '.join(a.gpus)}")
    if a.dry_run:
        for exp_id, cmd in todo:
            print(" ".join(shlex.quote(c) for c in cmd))
        return

    os.makedirs(a.out_dir, exist_ok=True)
    free, running, failed = list(a.gpus), {}, []
    while todo or running:
        while todo and free:
            exp_id, cmd = todo.pop(0)
            gpu = free.pop(0)
            log = open(os.path.join(a.out_dir, f"{exp_id}.log"), "w", encoding="utf-8")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            running[gpu] = (exp_id, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log, time.time())
            print(f"[{time.strftime('%H:%M')}] start {exp_id} on GPU {gpu}", flush=True)
        time.sleep(10)
        for gpu, (exp_id, proc, log, t0) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[gpu]
            free.append(gpu)
            status = "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode}, see {log.name})"
            if proc.returncode:
                failed.append(exp_id)
            print(f"[{time.strftime('%H:%M')}] {status} {exp_id} on GPU {gpu} after {(time.time() - t0) / 60:.0f} min",
                  flush=True)
    print("all jobs finished" + (f"; failed: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
