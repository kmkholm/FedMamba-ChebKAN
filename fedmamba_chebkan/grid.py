"""Fixed experiment grid for the R3 revision, executed sequentially (one fresh process per run, finished runs skipped).

python grid.py --list          # print the job list with an ETA
python grid.py                 # run everything not yet done
Order: 3 seeds of every experiment first, then the remaining seeds of the main and ablation experiments,
so a complete picture exists early and is then refined.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("FMCK_RESULTS", os.path.join(os.path.dirname(HERE), "results"))
LOG = os.path.join(RESULTS, "queue_log.txt")
MINUTES = {"fl": {"cwru": 12, "paderborn": 12, "mimii": 10, "cmapss": 15, "dirichlet": 14}, "pooled": 5, "cnn": 4}


def main_jobs(seeds):
    jobs = []
    for s in seeds:
        for ds in ("cwru", "paderborn", "mimii", "cmapss"):
            for m in ("spectral", "fedavg", "fedprox", "scaffold", "fedbn", "central", "local"):
                jobs.append(("main", ds, m, s, ["--save_model"] if (m == "spectral" and s == 0) else []))
            jobs.append(("main", ds, "fedavg", s, ["--backbone", "cnn", "--head", "linear"]))          # ablation A
    return jobs


def ablation_jobs(seeds):
    jobs = []
    for s in seeds:
        for ds in ("cwru", "mimii", "cmapss"):
            jobs.append(("ablation", ds, "fedavg", s, ["--head", "linear"]))                       # B
            jobs.append(("ablation", ds, "spectral", s, ["--tau", "0", "--tag", "_tau0"]))          # D
            jobs.append(("ablation", ds, "spectral", s, ["--quant8", "--tag", "_q8"]))              # F
        jobs.append(("ablation", "cwru", "fedavg", s, ["--quant8", "--tag", "_q8"]))                # 8-bit FedAvg (communication)
    return jobs


def dp_jobs(seeds):
    return [("dp", "cwru", "spectral", s, ["--dp_sigma", str(sig), "--tag", f"_dp{sig}"])
            for s in seeds for sig in (0.5, 0.8, 1.0, 1.5, 2.0)]


def sens_jobs(seeds):
    jobs = []
    for s in seeds:
        for tau in ("0.0001", "0.001", "0.01", "0"):
            for N in (4, 8, 12):
                if N == 8 and tau in ("0.001", "0"):
                    continue                                    # identical to main / ablation-D runs
                jobs.append(("sens", "cwru", "spectral", s, ["--tau", tau, "--degree", str(N), "--tag", f"_tau{tau}_N{N}"]))
    return jobs


def noniid_jobs(seeds):
    return [("noniid", "dirichlet", m, s, ["--alpha", str(a), "--tag", f"_a{a}"])
            for s in seeds for a in (0.05, 0.1, 0.3, 0.5, 1.0, -1) for m in ("fedavg", "fedprox", "scaffold", "spectral")]


def all_jobs():
    first, rest = [0, 1, 2], [3, 4, 5, 6, 7]
    return (main_jobs(first) + ablation_jobs(first) + dp_jobs(first) + sens_jobs(first) + noniid_jobs(first)
            + main_jobs(rest) + ablation_jobs(rest))


def name_of(job):
    exp, ds, m, s, extra = job
    bb = extra[extra.index("--backbone") + 1] if "--backbone" in extra else "mamba"
    hd = extra[extra.index("--head") + 1] if "--head" in extra else "kan"
    tag = extra[extra.index("--tag") + 1] if "--tag" in extra else ""
    return exp, f"{ds}_{m}_{bb}-{hd}{tag}_s{s}"


def minutes(job):
    exp, ds, m, s, extra = job
    if m in ("central", "local"):
        return MINUTES["pooled"]
    if "cnn" in extra:
        return MINUTES["cnn"]
    return MINUTES["fl"][ds]


def done(job):
    exp, name = name_of(job)
    return os.path.exists(os.path.join(RESULTS, exp, name + ".json"))


def log(msg):
    os.makedirs(RESULTS, exist_ok=True)
    line = time.strftime("%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    jobs = all_jobs()
    todo = [j for j in jobs if not done(j)]
    eta = sum(minutes(j) for j in todo)
    if a.list:
        print(f"{len(jobs)} jobs, {len(todo)} to do, ETA ~{eta / 60:.1f} h")
        for j in todo[:10]:
            print(" ", name_of(j))
        return
    log(f"queue start: {len(todo)} runs, ETA ~{eta / 60:.1f} h")
    for k, j in enumerate(todo):
        if done(j):
            continue
        exp, ds, m, s, extra = j
        cmd = [sys.executable, os.path.join(HERE, "run.py"), "--dataset", ds, "--method", m, "--seed", str(s), "--exp", exp] + extra
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        left = sum(minutes(x) for x in todo[k + 1:] if not done(x))
        log(f"[{k + 1}/{len(todo)}] {name_of(j)[1]} rc={r.returncode} {time.time() - t0:.0f}s | {tail} | left ~{left / 60:.1f} h")
        if r.returncode != 0:
            log("  stderr: " + r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "  (no stderr)")
    log("queue finished")


if __name__ == "__main__":
    main()
