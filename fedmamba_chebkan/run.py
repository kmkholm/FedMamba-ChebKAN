"""One experiment run -> results/<exp>/<name>.json   (skips runs that already exist).

python run.py --dataset cwru --method spectral --seed 0
methods: local | central | fedavg | fedprox | scaffold | fedbn | spectral | spectral_bn (SpectralFedAvg with client-local BatchNorm)
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D                                    # noqa: E402
from fl import run_federated, run_pooled            # noqa: E402
from models import Net                              # noqa: E402

SHAPES = {"cwru": (1, 1024, 16, 64), "paderborn": (1, 1024, 16, 64), "dirichlet": (1, 1024, 16, 64),
          "mimii": (64, 313, 8, 32), "cmapss": (14, 30, 1, 256)}          # c_in, length, patch, batch
RESULTS = os.environ.get("FMCK_RESULTS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", default="mamba")
    ap.add_argument("--head", default="kan")
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--degree", type=int, default=8)
    ap.add_argument("--quant8", action="store_true")
    ap.add_argument("--dp_sigma", type=float, default=0.0)
    ap.add_argument("--dp_clip", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=-1.0, help="Dirichlet alpha (dataset=dirichlet); <=0 means IID")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--exp", default="main")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--pooled_epochs", type=int, default=60)
    ap.add_argument("--scaffold_lr", type=float, default=0.05)
    a = ap.parse_args(argv)

    name = f"{a.dataset}_{a.method}_{a.backbone}-{a.head}{a.tag}_s{a.seed}"
    out_dir = os.path.join(RESULTS, a.exp)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".json")
    if os.path.exists(path):
        print("exists", path)
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c_in, length, patch, batch = SHAPES[a.dataset]
    task = D.TASK[a.dataset]
    if a.dataset == "dirichlet":
        clients = D.load("dirichlet", alpha=None if a.alpha <= 0 else a.alpha, seed=a.seed)
    else:
        clients = D.load(a.dataset)
    for c in clients:
        for k in list(c):
            if k.startswith(("x_", "y_")):
                c[k] = torch.as_tensor(c[k]).to(dev)
    cfg = dict(dataset=a.dataset, method=a.method, task=task, rounds=a.rounds, frac=0.3, dropout=0.075,
               local_epochs=2, batch=batch, lr=1e-3, wd=1e-4, mu=0.01, scaffold_lr=a.scaffold_lr, tau=a.tau, degree=a.degree,
               quant8=a.quant8, dp_sigma=a.dp_sigma, dp_clip=a.dp_clip, gamma=1.0, eval_every=5, device=dev,
               pooled_epochs=a.pooled_epochs, pooled_eval_every=2, backbone=a.backbone, head=a.head, alpha=a.alpha,
               clients=len(clients))

    def make_model():
        return Net(task, D.N_OUT[a.dataset], c_in, length, backbone=a.backbone, head=a.head,
                   degree=a.degree, patch=patch)

    print(f"== {name}  ({len(clients)} clients, task {task})", flush=True)
    t0 = time.time()
    if a.method in ("local", "central"):
        res, state = run_pooled(make_model, clients, cfg, a.seed, local_only=a.method == "local")
    else:
        res, state = run_federated(make_model, clients, cfg, a.seed)
    res.update(config={k: v for k, v in cfg.items() if k != "device"}, seed=a.seed, name=name,
               wall_s=round(time.time() - t0, 1), torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
               python=platform.python_version())
    if a.save_model and state is not None:
        torch.save(state, os.path.join(out_dir, name + ".pt"))
    with open(path + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(path + ".tmp", path)
    print(f"done {name}: primary {res['final']['primary']:.4f} in {res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
