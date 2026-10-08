"""Privacy study, final grid (CWRU, all non-overlapping training windows per client).
1) learning-rate check on seed 0 (3e-3 vs 5e-3 at sigma 8), chosen by the last-round VALIDATION accuracy;
2) seeds 1-3: no-noise reference and DP-SGD with sigma 4 / 8 / 15 -> results/dp_full/."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
P = os.path.join(RES, "public", "paderborn_pretrained.pt")
BASE = ["--dataset", "cwru", "--method", "spectral", "--norm", "group", "--win_norm", "--full_train", "--select_last",
        "--init", P]
DP = ["--batch", "4096", "--local_epochs", "4"]
LOG = os.path.join(RES, "dp_full_grid_log.txt")


def run(args):
    r = subprocess.run([sys.executable, os.path.join(HERE, "run.py")] + args, capture_output=True, text=True)
    tail = [l for l in r.stdout.splitlines() if l.startswith(("done", "exists"))]
    with open(LOG, "a") as f:
        f.write(" ".join(args[-6:]) + " | rc " + str(r.returncode) + " | " + (tail[-1] if tail else r.stderr[-300:]) + "\n")


def last_val(name):
    r = json.load(open(os.path.join(RES, "smoke", name + ".json")))
    return r["curve"][-1]["val"]


run(BASE + DP + ["--seed", "0", "--exp", "smoke", "--lr", "5e-3", "--dpsgd_sigma", "8", "--tag", "_full_dp8_lr5"])
v3 = last_val("cwru_spectral_mamba-kan_full_dp8_s0")
v5 = last_val("cwru_spectral_mamba-kan_full_dp8_lr5_s0")
lr = "5e-3" if v5 > v3 else "3e-3"
with open(LOG, "a") as f:
    f.write(f"validation (last round): lr 3e-3 {v3:.4f}, lr 5e-3 {v5:.4f} -> lr {lr}\n")
for s in ("1", "2", "3"):
    run(BASE + ["--seed", s, "--exp", "dp_full", "--tag", "_full_nodp"])
    for sg in ("4", "8", "15"):
        run(BASE + DP + ["--seed", s, "--exp", "dp_full", "--lr", lr, "--dpsgd_sigma", sg, "--tag", f"_full_dpsgd{sg}"])
with open(LOG, "a") as f:
    f.write("grid finished\n")
