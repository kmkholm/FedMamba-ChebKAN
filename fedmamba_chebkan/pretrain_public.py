"""Public pretraining for the privacy study: the Mamba encoder and first Chebyshev-KAN layer are trained on Paderborn
(a different public bearing test rig) and then fine-tuned with DP-SGD on the federated CWRU clients.

Source data: all healthy (K00x), outer-race (KAxx) and inner-race (KIxx) bearings, operating settings N09_M07_F10 and
N15_M07_F04 (neither is used by the federated Paderborn benchmark), vibration_1 decimated 64 kHz -> 12.8 kHz to match the
CWRU time scale, 1024-sample windows, every window standardised by its own mean and s.d. Bearings are split 80/20 into
training and validation; the epoch with the best validation accuracy is kept.

python pretrain_public.py   ->  results/public/paderborn_pretrained.pt
"""
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D                                    # noqa: E402
from models import Net                              # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "public")
SETTINGS = ["N09_M07_F10", "N15_M07_F04"]
FILES_PER_BEARING = 6
L = 1024


def label(b):
    return 0 if b.startswith("K0") else 1 if b.startswith("KA") else 2 if b.startswith("KI") else None


def load(frontend="raw"):
    rng = np.random.default_rng(D.PARTITION_SEED + 7)
    bearings = sorted(b for b in os.listdir(os.path.join(D.ROOT, "Paderborn")) if label(b) is not None)
    val_b = set()
    for c in range(3):                                  # 20 % of the bearings of every class for validation
        bc = [b for b in bearings if label(b) == c]
        val_b |= set(rng.choice(bc, max(1, round(0.2 * len(bc))), replace=False).tolist())
    xs = {"tr": [], "va": []}
    ys = {"tr": [], "va": []}
    for b in bearings:
        for st in SETTINGS:
            files = sorted(glob.glob(os.path.join(D.ROOT, "Paderborn", b, f"{st}_{b}_*.mat")))
            for f in [files[j] for j in rng.permutation(len(files))[:FILES_PER_BEARING]]:
                v = scipy.signal.decimate(D._paderborn_vib(f), 5, zero_phase=True).astype(np.float32)
                w = D._windows(v, L, L)
                s = "va" if b in val_b else "tr"
                xs[s].append(w)
                ys[s].append(np.full(len(w), label(b), np.int64))
    out = {}
    for s in xs:
        x = np.concatenate(xs[s])[:, None, :]
        if frontend == "fft":                            # same fixed spectral front-end as run.py --frontend fft
            x = np.log1p(np.abs(np.fft.rfft(x, axis=-1))[..., 1:]).astype(np.float32)
        x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)
        out[f"x_{s}"], out[f"y_{s}"] = torch.as_tensor(x), torch.as_tensor(np.concatenate(ys[s]))
    out["val_bearings"] = sorted(val_b)
    return out


def main(epochs=30, batch=128, seed=0, frontend="raw", d_model=64, layers=3, tag=""):
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    d = load(frontend)
    print(f"public data: {len(d['y_tr'])} train / {len(d['y_va'])} val windows ({time.time() - t0:.0f}s)", flush=True)
    model = Net("cls", 3, 1, L // 2 if frontend == "fft" else L, patch=16, norm="group", d_model=d_model, n_layers=layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xtr, ytr = d["x_tr"].to(dev), d["y_tr"].to(dev)
    xva, yva = d["x_va"].to(dev), d["y_va"].to(dev)
    best, log = (-1, None, 0), []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(ytr), device=dev)
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float(torch.cat([model(xva[j:j + 512]).argmax(1) for j in range(0, len(yva), 512)]).eq(yva).float().mean())
        log.append({"epoch": ep, "val_acc": acc})
        print(f"  epoch {ep:2d} val acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if acc > best[0]:
            best = (acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
    # transfer the encoder and the first Chebyshev-KAN layer; the output layer is task-specific
    keep = {k: v for k, v in best[1].items() if k.startswith(("enc.", "kan1."))}
    torch.save(keep, os.path.join(OUT, f"paderborn_pretrained{tag}.pt"))
    json.dump({"frontend": frontend, "d_model": d_model, "layers": layers, "settings": SETTINGS, "files_per_bearing": FILES_PER_BEARING, "decimation": 5, "window": L,
               "n_train": len(ytr), "n_val": len(yva), "val_bearings": d["val_bearings"], "best_epoch": best[2],
               "best_val_acc": best[0], "log": log}, open(os.path.join(OUT, f"paderborn_pretrained{tag}.json"), "w"), indent=1)
    print(f"saved encoder + kan1, best epoch {best[2]} val acc {best[0]:.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", default="raw", choices=["raw", "fft"])
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(frontend=a.frontend, d_model=a.d_model, layers=a.layers, tag=a.tag)
