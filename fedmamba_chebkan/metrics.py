"""Metrics. 'select' is the validation criterion (higher is better); 'primary' is the headline test metric."""
import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score, roc_curve


def phm08(d):
    d = np.asarray(d, dtype=np.float64)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def far_at_tpr(y, s, tpr_target=0.95):
    fpr, tpr, _ = roc_curve(y, s)
    i = np.searchsorted(tpr, tpr_target, side="left")
    return float(fpr[min(i, len(fpr) - 1)])


def evaluate(task, preds, ys, groups, split="te", full=False, dataset=""):
    p = preds.numpy()
    y = ys.numpy()
    out = {}
    if task == "cls":
        yhat = p.argmax(1)
        acc = float((yhat == y).mean())
        out.update(select=acc, primary=acc, acc=acc)
        if full:
            C = p.shape[1]
            prob = torch.softmax(torch.from_numpy(p), 1).numpy()
            out["macro_f1"] = float(f1_score(y, yhat, average="macro", labels=list(range(C)), zero_division=0))
            present = np.unique(y)
            if len(present) == C:
                out["macro_auroc"] = float(roc_auc_score(y, prob, multi_class="ovr", labels=list(range(C))))
            out["confusion"] = confusion_matrix(y, yhat, labels=list(range(C))).tolist()
        return out
    if task == "ae":
        if split != "te":
            v = -float(p.mean())
            return {"select": v, "primary": v}
        ids = np.array([g["machine_id"] for g in groups])
        snr = np.array([g["snr"] for g in groups])

        def block(mask):
            res = {"auroc": [], "auprc": [], "far95": []}
            for mid in np.unique(ids[mask]):
                m = mask & (ids == mid)
                if len(np.unique(y[m])) < 2:
                    continue
                res["auroc"].append(roc_auc_score(y[m], p[m]))
                res["auprc"].append(average_precision_score(y[m], p[m]))
                res["far95"].append(far_at_tpr(y[m], p[m]))
            return {k: float(np.mean(v)) for k, v in res.items()}

        main = block(snr == "0_dB")
        out.update(select=-float(p[y == 0].mean()), primary=main["auroc"], auroc=main["auroc"],
                   auprc=main["auprc"], far95=main["far95"])
        if full:
            for s in ("6_dB", "0_dB", "-6_dB"):
                out[f"by_snr_{s}"] = block(snr == s)
        return out
    # rul
    if split != "te":
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        return {"select": -rmse, "primary": rmse, "rmse": rmse}
    fd = np.array([g["fd"] for g in groups])
    rm, sc = {}, {}
    for f in np.unique(fd):
        m = fd == f
        rm[int(f)] = float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
        sc[int(f)] = phm08(p[m] - y[m])
    rmse = float(np.mean(list(rm.values())))
    out.update(select=-rmse, primary=rmse, rmse=rmse, score=float(np.mean(list(sc.values()))),
               rmse_by_fd=rm, score_by_fd=sc)
    if full:
        out["pred"] = p.round(2).tolist()
        out["true"] = y.round(2).tolist()
        out["fd"] = fd.tolist()
    return out
