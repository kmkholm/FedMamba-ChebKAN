"""Explainability of the trained SpectralFedAvg-BN model on real CWRU test windows (Fig 8 and Fig 9).

Fig 8: (a) integrated-gradients attribution on one correctly classified test window per class; (b) frequency-band occlusion:
       drop in the probability of the true class when a 250 Hz band is removed from the input, averaged per class.
Fig 9: (a) t-SNE of the encoder embeddings of all test windows (client-local BatchNorm), coloured by class, marker = sensor;
       (b) tail energy of the Chebyshev coefficients versus order, with the truncation tolerance tau;
       (c) learned Chebyshev-KAN output-edge functions (three strongest input edges per class).
python xai.py  ->  results/xai/Fig8.*, Fig9.*, xai_summary.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D                                    # noqa: E402
from models import Net                              # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "xai")
MODEL = os.path.join(RES, "cwru_spectral_bn_mamba-kan_s0.pt")
CLS = D.CWRU_CLASSES
CCOL = ["#3D5A80", "#E4572E", "#2A9D8F", "#B8860B"]
FS = 12000.0
dev = "cuda" if torch.cuda.is_available() else "cpu"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def save(fig, name):
    from PIL import Image
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RES, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    Image.open(os.path.join(RES, f"{name}.png")).convert("RGB").save(os.path.join(RES, f"{name}.tif"), dpi=(300, 300),
                                                                      compression="tiff_lzw")


def model_for(ck, i):
    m = Net("cls", 4, 1, 1024, patch=16).to(dev)
    m.load_state_dict(ck["global"])
    m.load_state_dict(ck["local_bn"][i], strict=False)
    return m.eval()


def integrated_gradients(m, x, target, steps=64):
    alphas = torch.linspace(0, 1, steps, device=dev).view(-1, 1, 1)
    xs = (alphas * x).requires_grad_(True)                      # baseline: zero signal
    out = m(xs)[:, target].sum()
    g, = torch.autograd.grad(out, xs)
    return (x * g.mean(0, keepdim=True)).squeeze().detach().cpu().numpy()


@torch.no_grad()
def band_occlusion(m, x, y, width=250.0):
    """Mean drop in p(true class) when the band [f, f+width) is removed (FFT zeroing), per band."""
    edges = np.arange(0, FS / 2, width)
    freqs = np.fft.rfftfreq(x.shape[-1], 1 / FS)
    p0 = F.softmax(m(x), 1).gather(1, y[:, None]).squeeze(1)
    X = torch.fft.rfft(x, dim=-1)
    drops = []
    for f0 in edges:
        mask = torch.as_tensor(~((freqs >= f0) & (freqs < f0 + width)), device=dev, dtype=X.dtype)
        xo = torch.fft.irfft(X * mask, n=x.shape[-1], dim=-1)
        p = F.softmax(m(xo), 1).gather(1, y[:, None]).squeeze(1)
        drops.append((p0 - p).cpu().numpy())
    return edges + width / 2, np.stack(drops, 1)                # (n, bands)


def edge_function(layer, i, j, t):
    tt = torch.as_tensor(t, dtype=torch.float32)
    T = layer.basis(tt[:, None]).squeeze(1)                    # (len, N+1)
    c = layer.coef[i, j].detach().cpu()
    return (layer.base[j, i].detach().cpu() * F.silu(tt) + T @ c).numpy()


def main():
    plt = _plt()
    ck = torch.load(MODEL, map_location=dev)
    clients = D.load("cwru")
    summ = {}

    # ---------------- Fig 8 (a): integrated gradients on one real window per class (load 1, drive-end client)
    ci = 3
    c = clients[ci]
    m = model_for(ck, ci)
    xte = torch.as_tensor(c["x_te"]).to(dev)
    yte = torch.as_tensor(c["y_te"]).to(dev)
    with torch.no_grad():
        pred = m(xte).argmax(1)
    fig = plt.figure(figsize=(7.4, 5.2))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.35, 1], wspace=0.28, hspace=0.55)
    tms = np.arange(1024) / FS * 1000
    for k in range(4):
        idx = int(((yte == k) & (pred == k)).nonzero()[0])
        a = integrated_gradients(m, xte[idx:idx + 1], k)
        sig = xte[idx, 0].cpu().numpy()
        rel = np.convolve(np.abs(a), np.ones(16) / 16, mode="same")
        rel = rel / (rel.max() + 1e-12)
        ax = fig.add_subplot(gs[k, 0])
        lim = np.abs(sig).max() * 1.08
        ax.imshow(rel[None, :], aspect="auto", cmap="Reds", vmin=0, vmax=1, alpha=0.75,
                  extent=[tms[0], tms[-1], -lim, lim], interpolation="bilinear", zorder=0)
        ax.plot(tms, sig, color="#1C1B22", lw=0.55, zorder=2)
        ax.set_ylim(-lim, lim)
        ax.set_ylabel(CLS[k], fontsize=8.5, color=CCOL[k], fontweight="bold")
        ax.set_yticks([])
        if k < 3:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Time (ms)")
        if k == 0:
            ax.set_title("(a) Integrated gradients on test windows", fontsize=9.5, fontweight="bold", loc="left")
        summ[f"ig_top10pct_share_{CLS[k]}"] = float(np.sort(np.abs(a))[-102:].sum() / (np.abs(a).sum() + 1e-12))
    # ---------------- Fig 8 (b): frequency-band occlusion over the drive-end and fan-end clients
    cent, allr = None, {k: [] for k in range(4)}
    for i, cl in enumerate(clients):
        if cl["meta"]["sensor"] == "BA":
            continue
        mi = model_for(ck, i)
        x = torch.as_tensor(cl["x_te"]).to(dev)
        y = torch.as_tensor(cl["y_te"]).to(dev)
        with torch.no_grad():
            ok = (mi(x).argmax(1) == y)
        cent, dr = band_occlusion(mi, x[ok], y[ok])
        for k in range(4):
            sel = (y[ok] == k).cpu().numpy()
            if sel.any():
                allr[k].append(dr[sel])
    ax = fig.add_subplot(gs[:, 1])
    for k in range(4):
        r = np.concatenate(allr[k]).mean(0)
        ax.plot(cent / 1000, r, color=CCOL[k], lw=1.6, marker="o", ms=2.5, label=CLS[k])
        summ[f"occlusion_peak_kHz_{CLS[k]}"] = float(cent[r.argmax()] / 1000)
        summ[f"occlusion_peak_drop_{CLS[k]}"] = float(r.max())
    ax.set_xlabel("Removed band centre (kHz)")
    ax.set_ylabel("Drop in p(true class)")
    ax.set_title("(b) Frequency-band occlusion", fontsize=9.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    save(fig, "Fig8")
    plt.close(fig)

    # ---------------- Fig 9 (a): t-SNE of encoder embeddings (client-local BatchNorm)
    from sklearn.manifold import TSNE
    feats, ys, sens = [], [], []
    with torch.no_grad():
        for i, cl in enumerate(clients):
            mi = model_for(ck, i)
            x = torch.as_tensor(cl["x_te"]).to(dev)
            feats.append(mi.enc(x).cpu().numpy())
            ys.append(cl["y_te"])
            sens += [cl["meta"]["sensor"]] * len(cl["y_te"])
    Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(np.concatenate(feats))
    ys, sens = np.concatenate(ys), np.array(sens)
    fig = plt.figure(figsize=(7.4, 5.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.62, wspace=0.55)
    ax = fig.add_subplot(gs[0, :2])
    for k in range(4):
        for s_, mk in (("DE", "o"), ("FE", "^"), ("BA", "s")):
            sel = (ys == k) & (sens == s_)
            ax.scatter(Z[sel, 0], Z[sel, 1], s=5, marker=mk, color=CCOL[k], alpha=0.65, lw=0)
    for k in range(4):
        ax.scatter([], [], color=CCOL[k], s=14, label=CLS[k])
    for s_, mk in (("DE", "o"), ("FE", "^"), ("BA", "s")):
        ax.scatter([], [], color="#555", marker=mk, s=14, label=s_)
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              handletextpad=0.2, columnspacing=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a) t-SNE of encoder embeddings", fontsize=9.5, fontweight="bold", loc="left")
    # ---------------- Fig 9 (c): tail energy of the Chebyshev coefficients versus order
    m0 = model_for(ck, 0)
    ax = fig.add_subplot(gs[0, 2:])
    for name, layer, col in (("Layer 1 (64→32)", m0.kan1, "#3D5A80"), ("Layer 2 (32→4)", m0.kan2, "#E4572E")):
        e = layer.coef.detach().cpu().pow(2)
        tail = torch.flip(torch.cumsum(torch.flip(e, [-1]), -1), [-1]) / e.sum(-1, keepdim=True)   # sum_{m>=n} / total
        med = tail.flatten(0, 1).median(0).values.numpy()
        q1, q3 = np.quantile(tail.flatten(0, 1).numpy(), [0.25, 0.75], axis=0)
        n = np.arange(len(med))
        ax.semilogy(n[1:], med[1:], color=col, lw=1.6, marker="o", ms=3, label=name)
        ax.fill_between(n[1:], q1[1:], q3[1:], color=col, alpha=0.15, lw=0)
        summ[f"median_tail_energy_from_order8_{name[:7]}"] = float(med[-1])
    ax.axhline(1e-3, color="#1C1B22", ls="--", lw=0.9)
    ax.text(1.1, 1.25e-3, r"$\tau=10^{-3}$", fontsize=8)
    ax.set_xlabel("Chebyshev order $n$")
    ax.set_ylabel("Tail energy")
    ax.set_title("(b) Coefficient tail energy", fontsize=9.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    # ---------------- Fig 9 (b): learned output-edge functions, three strongest inputs per class
    t = np.linspace(-1, 1, 200)
    for k in range(4):
        ax = fig.add_subplot(gs[1, k])
        curves = [edge_function(m0.kan2, i, k, t) for i in range(m0.kan2.coef.shape[0])]
        top = np.argsort([-np.sqrt(np.mean(cv ** 2)) for cv in curves])[:3]
        for r_, i in enumerate(top):
            ax.plot(t, curves[i], color=CCOL[k], lw=1.4, alpha=[1, 0.7, 0.45][r_], label=f"input {i}")
        ax.axhline(0, color="#999", lw=0.5)
        ax.set_title(CLS[k], fontsize=8.5, color=CCOL[k], fontweight="bold")
        ax.set_xlabel("$t$")
        if k == 0:
            ax.set_ylabel(r"$\phi_{ij}(t)$")
            ax.text(-0.35, 1.22, "(c) Learned output-edge functions", transform=ax.transAxes, fontsize=9.5,
                    fontweight="bold")
        ax.legend(frameon=False, fontsize=6, loc="best")
    save(fig, "Fig9")
    plt.close(fig)
    json.dump(summ, open(os.path.join(RES, "xai_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
