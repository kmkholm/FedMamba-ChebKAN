"""Tables, statistics and figures from results/<exp>/*.json.

python analysis.py [--out ../analysis]
Writes: per_seed_results.csv (S1 Data), summary.json, tables_*.csv and Fig*.png/.tif/.pdf.
Statistics: paired two-sided Wilcoxon signed-rank over seeds (exact null distribution), rank-biserial correlation
r_rb = (W+ - W-) / (W+ + W-) with differences oriented so that positive favours SpectralFedAvg, and Benjamini-Hochberg
adjustment within each comparison family.
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("FMCK_RESULTS", os.path.join(os.path.dirname(HERE), "results"))
METHODS = ["local", "fedavg", "fedprox", "scaffold", "fedbn", "spectral", "spectral_bn", "central"]
LABEL = {"local": "Local-only", "fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD", "fedbn": "FedBN",
         "spectral": "SpectralFedAvg", "spectral_bn": "SpectralFedAvg-BN", "central": "Pooled reference", "cnn": "FedAvg, 1-D CNN"}
COL = {"local": "#9C97A6", "fedavg": "#3D5A80", "fedprox": "#0F6E72", "scaffold": "#C9A227", "fedbn": "#B05C9A",
       "spectral": "#E4572E", "spectral_bn": "#B8431F", "central": "#1C1B22", "cnn": "#7B8CA6"}
METRICS = {"cwru": [("acc", "Accuracy", 1), ("macro_f1", "Macro-F1", 1), ("macro_auroc", "Macro-AUROC", 1)],
           "paderborn": [("acc", "Accuracy", 1), ("macro_f1", "Macro-F1", 1), ("macro_auroc", "Macro-AUROC", 1)],
           "mimii": [("auroc", "AUROC", 1), ("auprc", "AUPRC", 1), ("far95", "FAR@95%TPR", -1)],
           "cmapss": [("rmse", "RMSE", -1), ("score", "PHM08 score", -1)]}
PRIMARY = [("cwru", "acc"), ("cwru", "macro_f1"), ("paderborn", "acc"), ("paderborn", "macro_f1"),
           ("mimii", "auroc"), ("mimii", "far95"), ("cmapss", "rmse"), ("cmapss", "score")]


def load():
    rows = []
    for p in glob.glob(os.path.join(RESULTS, "*", "*.json")):
        exp = os.path.basename(os.path.dirname(p))
        if exp == "smoke":
            continue
        r = json.load(open(p))
        c = r["config"]
        rows.append({"exp": exp, "dataset": c["dataset"], "method": c["method"], "backbone": c["backbone"], "head": c["head"],
                     "tau": c["tau"], "degree": c["degree"], "quant8": c["quant8"], "dp_sigma": c["dp_sigma"],
                     "alpha": c.get("alpha", -1), "seed": r["seed"], "final": r["final"], "curve": r.get("curve", []),
                     "payload": r.get("payload_bytes_per_client_round"), "n_params": r.get("n_params"),
                     "best_round": r.get("best_round"), "wall_s": r.get("wall_s"), "name": r["name"],
                     "degree_hist": r.get("degree_hist")})
    return rows


def select(rows, **kw):
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            out.append(r)
    return sorted(out, key=lambda r: r["seed"])


def main_runs(rows, ds, m):
    if m == "cnn":
        return select(rows, exp="main", dataset=ds, method="fedavg", backbone="cnn")
    return select(rows, exp="main", dataset=ds, method=m, backbone="mamba", head="kan")


def vals(runs, key):
    return np.array([r["final"].get(key, np.nan) for r in runs], dtype=float)


def ms(v, pct=False, nd=3):
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "—"
    f = 100 if pct else 1
    d = 1 if pct else nd
    return f"{f * v.mean():.{d}f} ± {f * v.std(ddof=1) if len(v) > 1 else 0:.{d}f}"


# ------------------------------------------------------------------ statistics
def wilcoxon_exact(d):
    """Two-sided exact Wilcoxon signed-rank on paired differences d (zeros dropped, average ranks for ties).
    The null distribution is enumerated over all 2^n sign patterns of the observed ranks (exact with ties)."""
    d = np.asarray(d, float)
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0, 0.0, 0.0, 0, 0.0
    from scipy.stats import rankdata
    rk = rankdata(np.abs(d))
    wp = rk[d > 0].sum()
    wm = rk[d < 0].sum()
    tot = rk.sum()
    obs = min(wp, wm)
    # enumerate sign patterns
    count = 0
    for mask in range(1 << n):
        s = sum(rk[i] for i in range(n) if mask >> i & 1)
        if min(s, tot - s) <= obs + 1e-9:
            count += 1
    p = count / (1 << n)
    rrb = (wp - wm) / tot
    return float(min(p, 1.0)), float(wp), float(wm), n, float(rrb)


def bh(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * m / rank)
        q[i] = prev
    return q


def significance(rows):
    out = []
    for prop in ("spectral", "spectral_bn"):
        for ds, key in PRIMARY:
            sign = dict((k, sg) for k, _, sg in METRICS[ds])[key]
            ours = {r["seed"]: r["final"].get(key) for r in main_runs(rows, ds, prop)}
            for base in ("fedavg", "fedprox", "scaffold", "fedbn", "central"):
                b = {r["seed"]: r["final"].get(key) for r in main_runs(rows, ds, base)}
                seeds = sorted(set(ours) & set(b))
                if len(seeds) < 3:
                    continue
                d = sign * (np.array([ours[x] for x in seeds]) - np.array([b[x] for x in seeds]))
                p, wp, wm, n, rrb = wilcoxon_exact(d)
                out.append({"method": prop, "dataset": ds, "metric": key, "vs": base, "n": len(seeds),
                            "mean_diff": float(d.mean()), "p": p, "r_rb": rrb, "W_plus": wp, "W_minus": wm})
    for prop in ("spectral", "spectral_bn"):
        for fam in ("baselines", "central"):
            idx = [i for i, r in enumerate(out) if r["method"] == prop and (r["vs"] == "central") == (fam == "central")]
            if idx:
                q = bh([out[i]["p"] for i in idx])
                for i, qi in zip(idx, q):
                    out[i]["q"] = float(qi)
    return out


# ------------------------------------------------------------------ figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def save(fig, out, name):
    from PIL import Image
    png = os.path.join(out, name + ".png")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight", facecolor="white")
    im = Image.open(png).convert("RGB")
    if im.size[0] > 2250:
        im = im.resize((2250, round(im.size[1] * 2250 / im.size[0])), Image.LANCZOS)
    im.save(os.path.join(out, name + ".tif"), compression="tiff_lzw", dpi=(300, 300))


def fig_main(rows, out):
    plt = _plt()
    panels = [("cwru", "acc", "CWRU accuracy", 1, True), ("paderborn", "macro_f1", "Paderborn macro-F1", 1, False),
              ("mimii", "auroc", "MIMII AUROC (0 dB)", 1, False), ("mimii", "far95", "MIMII FAR@95% TPR", -1, False),
              ("cmapss", "rmse", "C-MAPSS RMSE", -1, False), ("cmapss", "score", "C-MAPSS PHM08 score", -1, False)]
    order = ["local", "fedavg", "fedprox", "scaffold", "fedbn", "spectral", "spectral_bn", "central"]
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.8))
    ypos = list(range(len(order)))[::-1]
    for ax, (ds, key, title, sign, pct) in zip(axes.flat, panels):
        lo, hi = np.inf, -np.inf
        for m, y in zip(order, ypos):
            v = vals(main_runs(rows, ds, m), key)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                continue
            f = 100 if pct else 1
            mu, sd = f * v.mean(), f * (v.std(ddof=1) if len(v) > 1 else 0)
            ax.errorbar(mu, y, xerr=sd, fmt="D" if m.startswith("spectral") else "o", ms=7 if m.startswith("spectral") else 6,
                        color=COL[m], mfc="white" if m == "central" else COL[m], mew=1.6, elinewidth=1.6, capsize=3)
            lo, hi = min(lo, mu - sd), max(hi, mu + sd)
        ax.set_yticks(ypos)
        ax.set_yticklabels([LABEL[m] for m in order] if ax in axes[:, 0] else [], fontsize=10)
        ax.set_title(title + (" (%)" if pct else "") + ("  ↑" if sign > 0 else "  ↓"), fontsize=11, fontweight="bold", loc="left")
        ax.axhline(0.5, color="#CFC9D6", lw=0.8, ls="--")
        ax.grid(axis="x", color="#ECE9F0", lw=0.7)
        if key == "score":
            ax.set_xscale("log")                       # SCAFFOLD diverges by orders of magnitude on C-MAPSS
        elif np.isfinite(lo):
            pad = 0.08 * (hi - lo + 1e-9)
            ax.set_xlim(lo - pad, hi + pad)
    fig.tight_layout(w_pad=1.0, h_pad=1.6)
    save(fig, out, "Fig2")
    plt.close(fig)


def fig_confusion(rows, out):
    plt = _plt()
    names = {"cwru": ["Normal", "Ball", "Inner", "Outer"], "paderborn": ["Healthy", "Outer", "Inner"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    for ax, ds in zip(axes, ("cwru", "paderborn")):
        runs = main_runs(rows, ds, "spectral_bn")
        if not runs:
            continue
        cm = np.sum([np.array(r["final"]["confusion"]) for r in runs], 0).astype(float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Oranges", vmin=0, vmax=1)
        k = len(names[ds])
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=9,
                        color="white" if cmn[i, j] > 0.6 else "#1C1B22")
        ax.set_xticks(range(k), names[ds])
        ax.set_yticks(range(k), names[ds])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{'CWRU' if ds == 'cwru' else 'Paderborn'} (all seeds pooled)", fontsize=11, fontweight="bold", loc="left")
    fig.tight_layout()
    save(fig, out, "Fig3")
    plt.close(fig)


def fig_rul(rows, out):
    plt = _plt()
    runs = main_runs(rows, "cmapss", "spectral")
    if not runs:
        return
    r0 = runs[0]["final"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})
    fcol = {1: "#3D5A80", 2: "#0F6E72", 3: "#C9A227", 4: "#B05C9A"}
    p, t, fd = np.array(r0["pred"]), np.array(r0["true"]), np.array(r0["fd"])
    for f in (1, 2, 3, 4):
        m = fd == f
        a1.scatter(t[m], p[m], s=10, color=fcol[f], alpha=0.7, label=f"FD00{f}")
    a1.plot([0, 125], [0, 125], color="#1C1B22", lw=1)
    a1.fill_between([0, 125], [-13, 112], [13, 138], color="#ECE9F0", zorder=0)
    a1.set_xlabel("True RUL (cycles)")
    a1.set_ylabel("Predicted RUL")
    a1.legend(fontsize=8, frameon=False)
    a1.set_title("Test engines, seed 0", fontsize=11, fontweight="bold", loc="left")
    meths = ["fedavg", "fedprox", "fedbn", "spectral", "spectral_bn", "central"]   # SCAFFOLD diverged (Table 4)
    w = 0.13
    for j, m in enumerate(meths):
        rr = main_runs(rows, "cmapss", m)
        if not rr:
            continue
        mu = [np.mean([x["final"]["rmse_by_fd"][str(f)] if str(f) in x["final"]["rmse_by_fd"] else x["final"]["rmse_by_fd"][f]
                       for x in rr]) for f in (1, 2, 3, 4)]
        a2.bar(np.arange(4) + (j - 2.5) * w, mu, w, color="white" if m == "central" else COL[m], edgecolor=COL[m], label=LABEL[m])
    a2.set_xticks(range(4), [f"FD00{f}" for f in (1, 2, 3, 4)])
    a2.set_ylabel("RMSE (cycles)")
    a2.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    a2.set_title("Per-subset RMSE (mean over seeds)", fontsize=11, fontweight="bold", loc="left")
    fig.tight_layout()
    save(fig, out, "Fig4")
    plt.close(fig)


def fig_noniid(rows, out):
    plt = _plt()
    alphas = [0.05, 0.1, 0.3, 0.5, 1.0, -1]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for m in ("fedavg", "fedprox", "scaffold", "spectral"):
        mu, sd = [], []
        for a in alphas:
            v = vals(select(rows, exp="noniid", method=m, alpha=a), "acc")
            mu.append(100 * v.mean() if len(v) else np.nan)
            sd.append(100 * v.std(ddof=1) if len(v) > 1 else 0)
        x = np.arange(len(alphas))
        ax.errorbar(x, mu, yerr=sd, marker="D" if m == "spectral" else "o", color=COL[m], label=LABEL[m], capsize=3, lw=1.6)
    ax.set_xticks(range(len(alphas)), ["0.05", "0.1", "0.3", "0.5", "1.0", "IID"])
    ax.set_xlabel("Dirichlet α (smaller = more heterogeneous)")
    ax.set_ylabel("CWRU accuracy (%)")
    ax.grid(color="#ECE9F0", lw=0.7)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, out, "Fig5")
    plt.close(fig)


def comm_table(rows):
    rows_out = []
    m_round = math.ceil(0.3 * 12)
    for lab, runs in [("FedAvg", main_runs(rows, "cwru", "fedavg")), ("FedProx", main_runs(rows, "cwru", "fedprox")),
                      ("SCAFFOLD", main_runs(rows, "cwru", "scaffold")), ("FedBN", main_runs(rows, "cwru", "fedbn")),
                      ("SpectralFedAvg", main_runs(rows, "cwru", "spectral")),
                      ("SpectralFedAvg-BN", main_runs(rows, "cwru", "spectral_bn"))]:
        if not runs:
            continue
        mb = np.mean([r["payload"] for r in runs]) / 1e6
        rows_out.append({"method": lab, "MB_per_client_round": mb, "uplink_GB_200": mb * m_round * 100 / 1e3,
                         "acc": float(np.mean(vals(runs, "acc"))), "acc_sd": float(np.std(vals(runs, "acc"), ddof=1)) if len(runs) > 1 else 0.0,
                         "n_params": runs[0]["n_params"], "seeds": len(runs)})
    return rows_out


def fig_comm(rows, out):
    plt = _plt()
    tab = comm_table(rows)
    if not tab:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 3.3))
    cmap = {"FedAvg": COL["fedavg"], "FedProx": COL["fedprox"], "SCAFFOLD": COL["scaffold"], "FedBN": COL["fedbn"],
            "SpectralFedAvg": COL["spectral"], "SpectralFedAvg-BN": COL["spectral_bn"]}
    for i, r in enumerate(tab[::-1]):
        q = "8-bit" in r["method"]
        a1.barh(i, 1000 * r["MB_per_client_round"], color="white" if q else cmap[r["method"]], edgecolor=cmap[r["method"]],
                hatch="///" if q else None, height=0.62)
        a1.text(1000 * r["MB_per_client_round"] * 1.02, i, f"{1000 * r['MB_per_client_round']:.0f}", va="center", fontsize=9)
    a1.set_yticks(range(len(tab)), [r["method"] for r in tab[::-1]], fontsize=9.5)
    a1.set_xlabel("kB per participating client per round")
    a1.set_title("Update size", fontsize=11, fontweight="bold", loc="left")
    for r in tab:
        q = "8-bit" in r["method"]
        a2.errorbar(1000 * r["uplink_GB_200"], 100 * r["acc"], yerr=100 * r["acc_sd"], fmt="D" if "Spectral" in r["method"] else "o",
                    color=cmap[r["method"]], mfc="white" if q else cmap[r["method"]], mew=1.6, capsize=3)
        a2.plot([], [], "D" if "Spectral" in r["method"] else "o", color=cmap[r["method"]], label=r["method"])
    a2.set_xlabel("Fleet uplink, 100 rounds (MB)")
    a2.set_ylabel("CWRU accuracy (%)")
    a2.grid(color="#ECE9F0", lw=0.7)
    a2.set_title("Accuracy vs. uplink", fontsize=11, fontweight="bold", loc="left")
    a2.legend(fontsize=8, frameon=False, loc="center right")
    fig.tight_layout(w_pad=2)
    save(fig, out, "Fig6")
    plt.close(fig)


def ablation_table(rows):
    """Component analysis from the main runs only: A (CNN, FedAvg), C (FedAvg), E (SpectralFedAvg), E-BN."""
    spec = [("C", "FedAvg", "fedavg"), ("E", "SpectralFedAvg", "spectral"), ("E-BN", "SpectralFedAvg-BN", "spectral_bn")]
    cols = (("cwru", "acc", True), ("paderborn", "acc", True), ("mimii", "auroc", False), ("cmapss", "rmse", False))
    out = []
    for k, lab, m in spec:
        r = {"row": k, "config": lab}
        for ds, key, pct in cols:
            v = vals(main_runs(rows, ds, m), key)
            r[f"{ds}_{key}"] = ms(v, pct=pct, nd=2 if ds == "cmapss" else 3)
            r[f"_{ds}"] = float(np.nanmean(v)) if len(v) else float("nan")
            r[f"_{ds}_sd"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
        out.append(r)
    return out


def fig_ablation(rows, out):
    plt = _plt()
    tab = ablation_table(rows)
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.8))
    cols = ["#3D5A80", "#E4572E", "#B8431F"]
    for ax, (ds, title, f) in zip(axes, [("cwru", "CWRU acc. (%)", 100), ("paderborn", "Paderborn acc. (%)", 100),
                                         ("mimii", "MIMII AUROC", 1), ("cmapss", "C-MAPSS RMSE (lower better)", 1)]):
        y = [f * r[f"_{ds}"] for r in tab]
        e = [f * r[f"_{ds}_sd"] for r in tab]
        ax.bar(range(len(tab)), y, yerr=e, color=cols, edgecolor="#1C1B22", lw=0.6, capsize=2)
        ax.set_xticks(range(len(tab)), [r["row"] for r in tab], fontsize=9)
        ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left")
        ok = [(v, er) for v, er in zip(y, e) if np.isfinite(v)]
        if ok:
            lo, hi = min(v - er for v, er in ok), max(v + er for v, er in ok)
            pad = 0.25 * (hi - lo + 1e-9)
            ax.set_ylim(lo - pad, hi + pad)
    fig.tight_layout()
    save(fig, out, "Fig7")
    plt.close(fig)


def fig_convergence(rows, out):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, (ds, title) in zip(axes, (("cwru", "CWRU test accuracy"), ("mimii", "MIMII test AUROC (0 dB)"))):
        for m in ("fedavg", "fedprox", "scaffold", "fedbn", "spectral", "spectral_bn"):
            runs = main_runs(rows, ds, m)
            if not runs:
                continue
            R = np.array([[c["round"] for c in r["curve"]] for r in runs][0])
            Y = np.array([[c["test"] for c in r["curve"]] for r in runs])
            mu, sd = Y.mean(0), Y.std(0)
            ax.plot(R, mu, color=COL[m], lw=1.8, label=LABEL[m])
            ax.fill_between(R, mu - sd, mu + sd, color=COL[m], alpha=0.15)
        ax.set_xlabel("Round")
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
        ax.grid(color="#ECE9F0", lw=0.7)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, out, "Fig5")
    plt.close(fig)


def dp_table(rows):
    try:
        import dp_accounting as dpa
    except ImportError:
        dpa = None
    out = []
    base = main_runs(rows, "cwru", "spectral")
    base = [r for r in base if r["seed"] in (0, 1, 2)]
    if base:
        out.append({"sigma": 0.0, "eps": None, "acc": ms(vals(base, "acc"), pct=True), "_acc": float(np.mean(vals(base, "acc")))})
    for sig in (2.0, 1.5, 1.0, 0.8, 0.5):
        runs = select(rows, exp="dp", dp_sigma=sig)
        eps = None
        if dpa is not None:
            acc = dpa.rdp.RdpAccountant(orders=[1 + x / 10 for x in range(1, 100)] + list(range(12, 256)))
            acc.compose(dpa.PoissonSampledDpEvent(0.3, dpa.GaussianDpEvent(sig)), 100)
            eps = float(acc.get_epsilon(1e-5))
        out.append({"sigma": sig, "eps": eps, "acc": ms(vals(runs, "acc"), pct=True),
                    "_acc": float(np.mean(vals(runs, "acc"))) if runs else None, "seeds": len(runs)})
    return out


def sens_table(rows):
    out = []
    for tau in (1e-4, 1e-3, 1e-2, 0.0):
        row = {"tau": tau}
        for N in (4, 8, 12):
            if N == 8 and tau == 1e-3:
                runs = [r for r in main_runs(rows, "cwru", "spectral") if r["seed"] in (0, 1, 2)]
            elif N == 8 and tau == 0.0:
                runs = [r for r in select(rows, exp="ablation", dataset="cwru", method="spectral", tau=0.0) if r["seed"] in (0, 1, 2)]
            else:
                runs = select(rows, exp="sens", tau=tau, degree=N)
            row[f"N{N}"] = ms(vals(runs, "acc"), pct=True)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "analysis"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load()
    print(len(rows), "result files")
    # per-seed CSV (S1 Data)
    keys = sorted({k for r in rows for k, v in r["final"].items() if isinstance(v, (int, float))})
    with open(os.path.join(a.out, "per_seed_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exp", "dataset", "method", "backbone", "head", "tau", "degree", "quant8", "dp_sigma", "alpha", "seed",
                    "payload_bytes_per_client_round", "n_params", "best_round", "wall_s"] + keys)
        for r in sorted([x for x in rows if x["exp"] == "main" and x["backbone"] == "mamba"],
                        key=lambda r: (r["exp"], r["dataset"], r["method"], r["name"])):
            w.writerow([r["exp"], r["dataset"], r["method"], r["backbone"], r["head"], r["tau"], r["degree"], r["quant8"],
                        r["dp_sigma"], r["alpha"], r["seed"], r["payload"], r["n_params"], r["best_round"], r["wall_s"]]
                       + [r["final"].get(k, "") for k in keys])
    summary = {"main": {}, "significance": significance(rows), "ablation": ablation_table(rows), "dp": dp_table(rows),
               "communication": comm_table(rows)}
    for ds, mets in METRICS.items():
        summary["main"][ds] = {}
        for m in METHODS + ["cnn"]:
            runs = main_runs(rows, ds, m)
            summary["main"][ds][m] = {"seeds": len(runs)}
            for k, _, _ in mets:
                v = vals(runs, k)
                summary["main"][ds][m][k] = ms(v, pct=(k == "acc"), nd=2 if ds == "cmapss" else 3)

    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    for fn in (fig_main, fig_confusion, fig_rul, fig_convergence, fig_comm, fig_ablation):
        try:
            fn(rows, a.out)
        except Exception as e:                      # figures need complete experiments; report and continue
            print("figure", fn.__name__, "skipped:", e)
    print(json.dumps(summary["main"], indent=1)[:3000])


if __name__ == "__main__":
    main()
