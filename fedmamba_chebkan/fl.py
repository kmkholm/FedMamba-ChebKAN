"""Federated training: Local-only, Centralised, FedAvg, FedProx, SCAFFOLD, FedBN, SpectralFedAvg
(+ optional 8-bit uploads and the matched-clipping Gaussian mechanism)."""
import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import evaluate

THROTTLE = os.environ.get("FMCK_THROTTLE", "")


def _pause():
    if THROTTLE and os.path.exists(THROTTLE):
        try:
            t = float(open(THROTTLE).read().strip() or 0)
            if t > 0:
                time.sleep(t)
        except (OSError, ValueError):
            pass


# ------------------------------------------------------------------ parameter groups
def bn_keys(model):
    keys = set()
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            for k in m.state_dict():
                keys.add(f"{name}.{k}")
    return keys


def cheb_keys(model):
    return {k for k in model.state_dict() if k.endswith(".coef")}


def float_keys(state):
    return [k for k, v in state.items() if v.dtype.is_floating_point]


# ------------------------------------------------------------------ local training
def loss_fn(task, out, y, x):
    if task == "cls":
        return F.cross_entropy(out, y)
    if task == "rul":
        return F.huber_loss(out, y, delta=13.0)
    return F.mse_loss(out, x)


def local_train(model, data, cfg, gen, global_state=None, prox_mu=0.0, scaffold=None, steps_out=None):
    """Train `model` in place for cfg.local_epochs on client data (tensors already on the device)."""
    x, y = data["x_tr"], data["y_tr"]
    n = len(x)
    bs = cfg["batch"]
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    gparams = None
    if prox_mu > 0:
        gparams = {k: v.detach().clone() for k, v in global_state.items()}
    model.train()
    steps = 0
    for _ in range(cfg["local_epochs"]):
        perm = torch.randperm(n, generator=gen, device="cpu").to(x.device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            out = model(x[idx])
            loss = loss_fn(cfg["task"], out, y[idx], x[idx])
            if gparams is not None:
                loss = loss + 0.5 * prox_mu * sum(((p - gparams[k]) ** 2).sum() for k, p in model.named_parameters())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if scaffold is not None:
                c, ck = scaffold
                for k, p in model.named_parameters():
                    if p.grad is not None:
                        p.grad.add_(c[k] - ck[k])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1
            _pause()
    if steps_out is not None:
        steps_out.append(steps)
    return model


# ------------------------------------------------------------------ SpectralFedAvg pieces
def tail_degrees(coef, tau):
    """coef: (in, out, N+1). Smallest degree N_k per edge with sum_{n>N_k} c_n^2 <= tau * ||c||^2."""
    e = coef.pow(2)
    tot = e.sum(-1, keepdim=True)
    tail = torch.flip(torch.cumsum(torch.flip(e, [-1]), -1), [-1])          # tail[n] = sum_{m>=n} c_m^2
    tail_after = torch.cat([tail[..., 1:], torch.zeros_like(tail[..., :1])], -1)  # sum_{m>n}
    ok = tail_after <= tau * tot + 1e-30
    # first n with ok=True (ok is monotone in n)
    return ok.float().argmax(-1)


def truncate(coef, deg):
    N = coef.shape[-1]
    mask = (torch.arange(N, device=coef.device).view(1, 1, N) <= deg.unsqueeze(-1)).to(coef.dtype)
    return coef * mask


def quantize8(t):
    s = t.abs().max().clamp(min=1e-12) / 127.0
    return torch.round(t / s).clamp(-127, 127) * s


# ------------------------------------------------------------------ the federated loop
def run_federated(make_model, clients, cfg, seed, log=print):
    """cfg keys: method, rounds, frac, dropout, local_epochs, batch, lr, wd, task, mu, tau, degree,
    quant8, dp_sigma, dp_clip, gamma, eval_every, device."""
    dev = cfg["device"]
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    method = cfg["method"]
    model = make_model().to(dev)
    gstate = {k: v.detach().clone() for k, v in model.state_dict().items()}
    K = len(clients)
    BN = bn_keys(model)
    CH = cheb_keys(model)
    fkeys = float_keys(gstate)
    n_params = sum(gstate[k].numel() for k in fkeys)
    dp = cfg.get("dp_sigma", 0) > 0
    # FedBN keeps BatchNorm layers on the client; under DP they also stay local, so no unprotected statistic is released
    local_bn = {i: {k: gstate[k].clone() for k in BN} for i in range(K)} if (method == "fedbn" or dp) else None
    scaf_c = {k: torch.zeros_like(v) for k, v in model.named_parameters()} if method == "scaffold" else None
    scaf_ck = {i: {k: torch.zeros_like(v) for k, v in model.named_parameters()} for i in range(K)} if method == "scaffold" else None
    beta = None
    if CH:
        N1 = cfg["degree"] + 1
        beta = (1.0 + torch.arange(N1, device=dev, dtype=torch.float32)) ** (-cfg.get("gamma", 1.0))
    m_per_round = max(1, math.ceil(cfg["frac"] * K))
    best = {"val": None, "state": None, "bn": None, "round": 0}
    curve, payload_rounds, degree_hist = [], [], []
    t0 = time.time()
    for rnd in range(1, cfg["rounds"] + 1):
        if dp:   # Poisson sampling at rate q = frac (needed by the subsampled-Gaussian accountant)
            sel = [i for i in range(K) if np_rng.random() < cfg["frac"]]
        else:
            sel = list(np_rng.choice(K, m_per_round, replace=False))
            sel = [i for i in sel if np_rng.random() >= cfg["dropout"]] or [sel[0]]
        uploads, weights, bytes_up = [], [], 0
        for i in sel:
            model.load_state_dict(gstate)
            if local_bn is not None:
                model.load_state_dict(local_bn[i], strict=False)
            steps = []
            local_train(model, clients[i], cfg, gen, global_state=gstate if method == "fedprox" else None,
                        prox_mu=cfg["mu"] if method == "fedprox" else 0.0,
                        scaffold=(scaf_c, scaf_ck[i]) if method == "scaffold" else None, steps_out=steps)
            st = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if local_bn is not None:
                local_bn[i] = {k: st[k].clone() for k in BN}
            up = {k: (st[k] - gstate[k]) for k in fkeys if not (local_bn is not None and k in BN)}
            nb_vals = sum(v.numel() for v in up.values())
            if method == "spectral":
                for k in CH:
                    deg = tail_degrees(st[k], cfg["tau"])
                    degree_hist.append(deg.flatten().cpu().numpy().astype(np.int8))
                    up[k] = truncate(st[k], deg) - gstate[k]          # truncated coefficient state, as a delta
                    nb_vals += int((deg + 1).sum().item()) - st[k].numel()
                    bytes_up += deg.numel()                           # one byte per transmitted degree
            if method == "scaffold":
                ck_new = {}
                lr_k = cfg["lr"] * max(steps[0], 1)
                for k, p in model.named_parameters():
                    ck_new[k] = scaf_ck[i][k] - scaf_c[k] + (gstate[k] - st[k]) / lr_k
                up["__dc__"] = {k: ck_new[k] - scaf_ck[i][k] for k in ck_new}
                scaf_ck[i] = ck_new
                nb_vals *= 2
            if cfg.get("quant8"):
                for k in list(up):
                    if k != "__dc__":
                        up[k] = quantize8(up[k])
                bytes_up += nb_vals                                   # 1 byte per value (+ per-tensor scale, negligible)
            else:
                bytes_up += 4 * nb_vals
            uploads.append(up)
            weights.append(len(clients[i]["y_tr"]))
        payload_rounds.append(bytes_up / max(len(sel), 1))
        if not uploads:
            continue
        # -------- aggregation
        if dp:
            # matched clipping: rescale Chebyshev order n by 1/beta_n, clip the joint update to S, sum, add
            # N(0, sigma^2 S^2) per coordinate, divide by the expected number of participants, rescale by beta_n
            S, sig = cfg["dp_clip"], cfg["dp_sigma"]
            keys = [k for k in uploads[0] if k != "__dc__"]
            agg = {k: torch.zeros_like(gstate[k]) for k in keys}
            for up in uploads:
                scaled = {k: (up[k] / beta if k in CH else up[k]) for k in keys}
                norm = torch.sqrt(sum(v.pow(2).sum() for v in scaled.values()))
                f = min(1.0, S / (norm.item() + 1e-12))
                for k in keys:
                    agg[k] += scaled[k] * f
            denom = cfg["frac"] * K
            for k in keys:
                agg[k] = (agg[k] + torch.randn_like(agg[k]) * sig * S) / denom
                if k in CH:
                    agg[k] = agg[k] * beta
                gstate[k] = gstate[k] + agg[k]
        else:
            w = torch.tensor(weights, dtype=torch.float32, device=dev)
            w = w / w.sum()
            for k in uploads[0]:
                if k == "__dc__":
                    continue
                gstate[k] = gstate[k] + sum(wi * up[k] for wi, up in zip(w, uploads))
            if method == "scaffold":
                for k in scaf_c:
                    scaf_c[k] += (len(uploads) / K) * sum(up["__dc__"][k] for up in uploads) / len(uploads)
        # -------- evaluation
        if rnd % cfg["eval_every"] == 0 or rnd == cfg["rounds"]:
            val = evaluate_global(model, gstate, local_bn, clients, cfg, split="va")
            tst = evaluate_global(model, gstate, local_bn, clients, cfg, split="te")
            curve.append({"round": rnd, "val": val["select"], "test": tst["primary"], "time_s": round(time.time() - t0, 1)})
            better = best["val"] is None or (val["select"] > best["val"])
            if better:
                best = {"val": val["select"], "state": {k: v.clone() for k, v in gstate.items()},
                        "bn": copy.deepcopy(local_bn), "round": rnd}
            log(f"  r{rnd:3d} val {val['select']:.4f} test {tst['primary']:.4f} ({time.time() - t0:.0f}s)")
    final = evaluate_global(model, best["state"], best["bn"], clients, cfg, split="te", full=True)
    out = {"final": final, "best_round": best["round"], "curve": curve, "n_params": n_params,
           "payload_bytes_per_client_round": float(np.mean(payload_rounds)) if payload_rounds else 0.0,
           "train_time_s": round(time.time() - t0, 1)}
    if degree_hist:
        h = np.concatenate(degree_hist)
        out["degree_hist"] = np.bincount(h, minlength=cfg["degree"] + 1).tolist()
    return out, best["state"]


@torch.no_grad()
def evaluate_global(model, state, local_bn, clients, cfg, split="te", full=False):
    preds, ys, groups, xs_err = [], [], [], []
    for i, c in enumerate(clients):
        model.load_state_dict(state)
        if local_bn is not None:
            model.load_state_dict(local_bn[i], strict=False)
        model.eval()
        x, y = c[f"x_{split}"], c[f"y_{split}"]
        if len(y) == 0:
            continue
        outs = []
        for j in range(0, len(x), 256):
            o = model(x[j:j + 256])
            if cfg["task"] == "ae":
                o = (o - x[j:j + 256]).pow(2).mean(dim=(1, 2))
            outs.append(o.float())
        preds.append(torch.nan_to_num(torch.cat(outs), nan=0.0, posinf=1e6, neginf=-1e6).cpu())
        ys.append(y.cpu())
        groups += [c["meta"]] * len(y)
    return evaluate(cfg["task"], torch.cat(preds), torch.cat(ys), groups, split=split, full=full, dataset=cfg["dataset"])


# ------------------------------------------------------------------ non-federated references
def run_pooled(make_model, clients, cfg, seed, local_only=False, log=print):
    """Centralised (pooled data of all clients) or Local-only (each client alone, mean over clients)."""
    dev = cfg["device"]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    groups = [list(range(len(clients)))] if not local_only else [[i] for i in range(len(clients))]
    results = []
    for g in groups:
        model = make_model().to(dev)
        data = {"x_tr": torch.cat([clients[i]["x_tr"] for i in g]), "y_tr": torch.cat([clients[i]["y_tr"] for i in g])}
        sub = [clients[i] for i in g]
        c2 = dict(cfg, local_epochs=1)
        best_v, best_s = None, None
        for ep in range(cfg["pooled_epochs"]):
            local_train(model, data, c2, gen)
            if (ep + 1) % cfg["pooled_eval_every"] == 0:
                st = {k: v.detach().clone() for k, v in model.state_dict().items()}
                v = evaluate_global(model, st, None, sub, cfg, split="va")["select"]
                if best_v is None or v > best_v:
                    best_v, best_s = v, st
        res = evaluate_global(model, best_s, None, clients, cfg, split="te", full=True)
        results.append(res)
        log(f"  pooled group {g[:3]}... test {res['primary']:.4f}")
    if local_only:
        keys = [k for k, v in results[0].items() if isinstance(v, (int, float))]
        final = {k: float(np.mean([r[k] for r in results])) for k in keys}
        final["per_client"] = [{k: r[k] for k in keys} for r in results]
    else:
        final = results[0]
    return {"final": final, "train_time_s": round(time.time() - t0, 1)}, None
