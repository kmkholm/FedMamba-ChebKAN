"""Federated client partitions for CWRU, Paderborn, MIMII (fan) and C-MAPSS.

Leakage control: every split is made on whole recordings (or engines) or on contiguous, non-overlapping
segments of a recording *before* windowing, so no window shares samples with a window in another split.
Normalisation statistics are fitted on each client's training windows only.
Each loader returns a list of client dicts with numpy arrays: x_tr, y_tr, x_va, y_va, x_te, y_te (+ meta).
Partitions are deterministic (PARTITION_SEED); training seeds vary only initialisation and sampling.
"""
import glob
import json
import math
import os

import numpy as np
import scipy.io as sio

ROOT = os.environ.get("FMCK_DATA", r"F:/Training_Data/FedMamba_ChebKAN")
MIMII_ROOT = os.environ.get("FMCK_MIMII", r"D:/Training_Data/FedMamba_ChebKAN/MIMII")
CACHE = os.path.join(ROOT, "cache")
PARTITION_SEED = 2026


def _windows(x, length, hop):
    n = 1 + (len(x) - length) // hop if len(x) >= length else 0
    if n <= 0:
        return np.zeros((0, length), np.float32)
    idx = np.arange(length)[None, :] + hop * np.arange(n)[:, None]
    return x[idx].astype(np.float32)


def _take(rng, arr, k):
    if len(arr) <= k:
        return arr
    return arr[np.sort(rng.choice(len(arr), k, replace=False))]


def _znorm(c):
    mu, sd = c["x_tr"].mean(), c["x_tr"].std() + 1e-8
    for s in ("x_tr", "x_va", "x_te"):
        c[s] = ((c[s] - mu) / sd).astype(np.float32)
    return c


# ------------------------------------------------------------------ CWRU
CWRU_FILES = {  # class -> list over loads 0..3 of file ids (12 kHz drive-end experiments, outer race at 6:00)
    0: [[97], [98], [99], [100]],                                                    # normal
    1: [[118, 185, 222], [119, 186, 223], [120, 187, 224], [121, 188, 225]],          # ball 7/14/21 mil
    2: [[105, 169, 209], [106, 170, 210], [107, 171, 211], [108, 172, 212]],          # inner race
    3: [[130, 197, 234], [131, 198, 235], [132, 199, 236], [133, 200, 237]],          # outer race @6
}
CWRU_CLASSES = ["Normal", "Ball", "Inner race", "Outer race"]


def _cwru_signal(fid, sensor):
    folder = "normal_baseline" if fid in (97, 98, 99, 100) else "12k_drive_end"
    m = sio.loadmat(os.path.join(ROOT, "CWRU", folder, f"{fid}.mat"))
    keys = [k for k in m if k.endswith(f"_{sensor}_time")]
    return m[keys[0]].ravel().astype(np.float32) if keys else None


def cwru_clients(n_train_per_class=65, n_eval_per_class=40, L=1024):
    """12 clients = 4 motor loads x 3 sensor positions (DE, FE, BA). Each recording is cut into contiguous
    60/20/20 % segments (train/val/test) before windowing (50 % overlap inside a segment only).
    The normal-baseline files carry no BA channel, so the three BA clients hold no Normal windows."""
    rng = np.random.default_rng(PARTITION_SEED)
    clients = []
    for load in range(4):
        for sensor in ("DE", "FE", "BA"):
            parts = {s: ([], []) for s in ("tr", "va", "te")}
            for cls, per_load in CWRU_FILES.items():
                segs = {s: [] for s in parts}
                for fid in per_load[load]:
                    x = _cwru_signal(fid, sensor)
                    if x is None:
                        continue
                    a, b = int(0.6 * len(x)), int(0.8 * len(x))
                    segs["tr"].append(_windows(x[:a], L, L // 2))
                    segs["va"].append(_windows(x[a:b], L, L // 2))
                    segs["te"].append(_windows(x[b:], L, L // 2))
                for s, cap in (("tr", n_train_per_class), ("va", n_eval_per_class), ("te", n_eval_per_class)):
                    if segs[s]:
                        w = _take(rng, np.concatenate(segs[s]), cap)
                        parts[s][0].append(w)
                        parts[s][1].append(np.full(len(w), cls, np.int64))
            c = {f"x_{s}": np.concatenate(parts[s][0])[:, None, :] for s in parts}
            c.update({f"y_{s}": np.concatenate(parts[s][1]) for s in parts})
            c["meta"] = {"client": f"load{load}_{sensor}", "load_hp": load, "sensor": sensor}
            clients.append(_znorm(c))
    return clients


def cwru_pooled_de(n_train_per_class=260, n_eval_per_class=160, L=1024):
    """Pooled drive-end data of all four loads, used by the Dirichlet non-IID study."""
    rng = np.random.default_rng(PARTITION_SEED + 1)
    out = {s: ([], []) for s in ("tr", "va", "te")}
    for cls, per_load in CWRU_FILES.items():
        segs = {s: [] for s in out}
        for load in range(4):
            for fid in per_load[load]:
                x = _cwru_signal(fid, "DE")
                a, b = int(0.6 * len(x)), int(0.8 * len(x))
                segs["tr"].append(_windows(x[:a], L, L // 2))
                segs["va"].append(_windows(x[a:b], L, L // 2))
                segs["te"].append(_windows(x[b:], L, L // 2))
        for s, cap in (("tr", n_train_per_class * 4), ("va", n_eval_per_class), ("te", n_eval_per_class)):
            w = _take(rng, np.concatenate(segs[s]), cap)
            out[s][0].append(w)
            out[s][1].append(np.full(len(w), cls, np.int64))
    d = {f"x_{s}": np.concatenate(out[s][0])[:, None, :] for s in out}
    d.update({f"y_{s}": np.concatenate(out[s][1]) for s in out})
    mu, sd = d["x_tr"].mean(), d["x_tr"].std() + 1e-8
    for s in ("x_tr", "x_va", "x_te"):
        d[s] = ((d[s] - mu) / sd).astype(np.float32)
    return d


def dirichlet_clients(alpha, K=12, seed=0):
    """Re-partition the pooled CWRU training windows over K clients with Dir(alpha) class proportions
    (alpha=None: IID). Validation and test sets are the pooled held-out segments, shared by all clients."""
    d = cwru_pooled_de()
    rng = np.random.default_rng(PARTITION_SEED + 100 + seed)
    y = d["y_tr"]
    idx_per_client = [[] for _ in range(K)]
    if alpha is None:
        perm = rng.permutation(len(y))
        for k, part in enumerate(np.array_split(perm, K)):
            idx_per_client[k] = list(part)
    else:
        for c in np.unique(y):
            ids = rng.permutation(np.where(y == c)[0])
            p = rng.dirichlet(alpha * np.ones(K))
            cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
            for k, part in enumerate(np.split(ids, cuts)):
                idx_per_client[k] += list(part)
    clients = []
    nva = len(d["y_va"]) // K
    for k in range(K):
        ii = np.array(sorted(idx_per_client[k]), dtype=int)
        if len(ii) < 2:  # keep every client non-empty
            ii = rng.choice(len(y), 2, replace=False)
        sl = slice(k * nva, (k + 1) * nva)
        te = slice(None) if k == 0 else slice(0, 0)        # the shared test set is evaluated once
        clients.append({"x_tr": d["x_tr"][ii], "y_tr": y[ii], "x_va": d["x_va"][sl], "y_va": d["y_va"][sl],
                        "x_te": d["x_te"][te], "y_te": d["y_te"][te], "meta": {"client": f"dir{k}", "alpha": alpha}})
    return clients


# ------------------------------------------------------------------ Paderborn
PADERBORN_BEARINGS = {"K001": 0, "K002": 0, "KA04": 1, "KA15": 1, "KI04": 2, "KI14": 2}  # healthy / OR / IR (real damage)
PADERBORN_SETTINGS = ["N15_M07_F10", "N15_M01_F10"]
PADERBORN_CLASSES = ["Healthy", "Outer race", "Inner race"]


def _paderborn_vib(path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    s = m[[k for k in m if not k.startswith("__")][0]]
    for ch in s.Y:
        if ch.Name == "vibration_1":
            return np.asarray(ch.Data, dtype=np.float32)
    raise KeyError("vibration_1")


def paderborn_clients(n_train=220, n_eval=120, L=1024):
    """12 clients; each holds one healthy, one outer-race and one inner-race bearing (real damage) under one operating
    setting, i.e. device and condition skew with all three classes present. Per setting, the 8 bearing combinations
    minus the two complementary ones (all-first, all-second) give 6 clients in which every bearing appears 3 times.
    The 20 recordings of a (bearing, setting) pair are dealt to its 3 clients without overlap (6 each:
    4 train / 1 val / 1 test), and windows are cut only after this split."""
    rng = np.random.default_rng(PARTITION_SEED + 2)
    by_cls = {c: [b for b, k in PADERBORN_BEARINGS.items() if k == c] for c in range(3)}
    combos = [(h, o, i) for h in range(2) for o in range(2) for i in range(2) if (h, o, i) not in ((0, 0, 0), (1, 1, 1))]
    clients = []
    for st in PADERBORN_SETTINGS:
        # deal recordings: for each bearing, a random order of its 20 files, consumed 6 at a time by its clients
        pools = {}
        for b in PADERBORN_BEARINGS:
            files = sorted(glob.glob(os.path.join(ROOT, "Paderborn", b, f"{st}_{b}_*.mat")),
                           key=lambda p: int(p.rsplit("_", 1)[1][:-4]))
            pools[b] = [files[j] for j in rng.permutation(len(files))]
        used = {b: 0 for b in PADERBORN_BEARINGS}
        for combo in combos:
            bearings = [by_cls[c][combo[c]] for c in range(3)]
            parts = {s: ([], []) for s in ("tr", "va", "te")}
            rec = {}
            for b in bearings:
                fs = pools[b][used[b]:used[b] + 6]
                used[b] += 6
                split = {"tr": fs[:4], "va": fs[4:5], "te": fs[5:6]}
                rec[b] = {s: [os.path.basename(f) for f in v] for s, v in split.items()}
                for s, v in split.items():
                    w = np.concatenate([_windows(_paderborn_vib(f), L, L // 2) for f in v])
                    w = _take(rng, w, (n_train if s == "tr" else n_eval) // 3)
                    parts[s][0].append(w)
                    parts[s][1].append(np.full(len(w), PADERBORN_BEARINGS[b], np.int64))
            c = {f"x_{s}": np.concatenate(parts[s][0])[:, None, :] for s in parts}
            c.update({f"y_{s}": np.concatenate(parts[s][1]) for s in parts})
            c["meta"] = {"client": f"{st}_{'-'.join(bearings)}", "setting": st, "bearings": bearings, "recordings": rec}
            clients.append(_znorm(c))
    return clients


# ------------------------------------------------------------------ MIMII (fan)
def _mel_filterbank(sr=16000, n_fft=1024, n_mels=64, fmin=0.0, fmax=8000.0):
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    return fb


def _logmel(wav, fb, n_fft=1024, hop=512):
    import scipy.signal
    _, _, Z = scipy.signal.stft(wav, nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft, boundary="even", padded=False)
    p = (np.abs(Z) ** 2).astype(np.float32)
    return 10.0 * np.log10(fb @ p + 1e-10)


def mimii_clients(n_train=215, n_val=40, n_test_normal=100, n_test_abnormal=100):
    """12 clients = machine IDs 00/02/04/06 x SNR (+6, 0, -6 dB). Clips are independent 10-s recordings, so the
    split is by clip: training and validation use normal clips only; the test set holds unseen normal and abnormal clips.
    Features: 64 x 313 log-Mel (channel 0, STFT 1024, hop 512), cached."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, "mimii_fan_logmel.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return list(z["clients"])
    import soundfile as sf
    fb = _mel_filterbank()
    rng = np.random.default_rng(PARTITION_SEED + 3)
    clients = []
    for mid in ("id_00", "id_02", "id_04", "id_06"):
        for snr in ("6_dB", "0_dB", "-6_dB"):
            base = os.path.join(MIMII_ROOT, snr, "fan", mid)
            nor = sorted(glob.glob(os.path.join(base, "normal", "*.wav")))
            ab = sorted(glob.glob(os.path.join(base, "abnormal", "*.wav")))
            nor = [nor[i] for i in rng.permutation(len(nor))]
            ab = [ab[i] for i in rng.permutation(len(ab))][:n_test_abnormal]
            tr, va, te_n = nor[:n_train], nor[n_train:n_train + n_val], nor[n_train + n_val:n_train + n_val + n_test_normal]

            def feats(paths):
                return np.stack([_logmel(sf.read(p)[0][:, 0].astype(np.float32), fb) for p in paths]).astype(np.float32)

            x_tr, x_va, x_te = feats(tr), feats(va), np.concatenate([feats(te_n), feats(ab)])
            c = {"x_tr": x_tr, "y_tr": np.zeros(len(x_tr), np.int64), "x_va": x_va, "y_va": np.zeros(len(x_va), np.int64),
                 "x_te": x_te, "y_te": np.r_[np.zeros(len(te_n)), np.ones(len(ab))].astype(np.int64),
                 "meta": {"client": f"{mid}_{snr}", "machine_id": mid, "snr": snr,
                          "files": {"train": [os.path.basename(p) for p in tr], "val": [os.path.basename(p) for p in va],
                                    "test_normal": [os.path.basename(p) for p in te_n],
                                    "test_abnormal": [os.path.basename(p) for p in ab]}}}
            mu = c["x_tr"].mean(axis=(0, 2), keepdims=True)
            sd = c["x_tr"].std(axis=(0, 2), keepdims=True) + 1e-6
            for s in ("x_tr", "x_va", "x_te"):
                c[s] = ((c[s] - mu) / sd).astype(np.float32)
            clients.append(c)
            print("mimii", c["meta"]["client"], x_tr.shape, flush=True)
    np.savez(cache, clients=np.array(clients, dtype=object))
    return clients


# ------------------------------------------------------------------ C-MAPSS
CMAPSS_SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
RUL_CAP = 125


def _cmapss_read(path):
    a = np.loadtxt(path)
    cond = np.round(a[:, 2]).astype(int) * 1000 + np.round(a[:, 4]).astype(int)   # altitude x throttle -> operating condition
    return a[:, 0].astype(int), cond, a[:, [4 + s for s in CMAPSS_SENSORS]].astype(np.float32)


def cmapss_clients(window=30, n_train=900, clients_per_fd=5):
    """20 clients = 5 groups of training engines per FD subset. Engines (not windows) are split: within a client,
    one engine in ten is held out for validation. Test: the official test engines of each subset, scored on the
    last window of each engine against the official RUL (one prediction per engine, as in the PHM08 protocol)."""
    rng = np.random.default_rng(PARTITION_SEED + 4)
    base = os.path.join(ROOT, "C-MAPSS", "CMAPSSData")
    clients = []
    for fd in range(1, 5):
        uid, cond, X = _cmapss_read(os.path.join(base, f"train_FD00{fd}.txt"))
        units = rng.permutation(np.unique(uid))
        tid, tcond, TX = _cmapss_read(os.path.join(base, f"test_FD00{fd}.txt"))
        true_rul = np.loadtxt(os.path.join(base, f"RUL_FD00{fd}.txt")).astype(np.float32)
        for g, group in enumerate(np.array_split(units, clients_per_fd)):
            va_units = group[::10]
            tr_units = np.setdiff1d(group, va_units)
            # operating-condition-wise standardisation (FD002/FD004 have six conditions), fitted on this client's
            # training engines only; an unseen condition falls back to the client's pooled statistics
            trm = np.isin(uid, tr_units)
            g_mu, g_sd = X[trm].mean(0), X[trm].std(0) + 1e-6
            stats = {c: (X[trm & (cond == c)].mean(0), X[trm & (cond == c)].std(0) + 1e-6)
                     for c in np.unique(cond[trm]) if (trm & (cond == c)).sum() > 10}

            def sc(a, cnd):
                out = np.empty_like(a)
                for i in range(len(a)):
                    mu, sd = stats.get(int(cnd[i]), (g_mu, g_sd))
                    out[i] = (a[i] - mu) / sd
                return np.clip(out, -6, 6).astype(np.float32)

            def engine_windows(units_, last_only=False):
                xs, ys = [], []
                for u in units_:
                    xu = sc(X[uid == u], cond[uid == u])
                    T = len(xu)
                    rul = np.minimum(np.arange(T)[::-1], RUL_CAP).astype(np.float32)
                    starts = [T - window] if last_only else range(0, T - window + 1)
                    for s0 in starts:
                        if s0 < 0:
                            continue
                        xs.append(xu[s0:s0 + window].T)
                        ys.append(rul[s0 + window - 1])
                return np.stack(xs), np.array(ys, np.float32)

            x_tr, y_tr = engine_windows(tr_units)
            sel = _take(rng, np.arange(len(x_tr)), n_train)
            x_va, y_va = engine_windows(va_units)
            vsel = _take(rng, np.arange(len(x_va)), 200)          # validation windows capped for evaluation speed
            x_va, y_va = x_va[vsel], y_va[vsel]
            # test: official test engines of this subset, assigned round-robin to the subset's clients
            t_units = np.unique(tid)[g::clients_per_fd]
            xt, yt = [], []
            for j, u in enumerate(t_units):
                xu = sc(TX[tid == u], tcond[tid == u])
                if len(xu) < window:
                    xu = np.vstack([np.repeat(xu[:1], window - len(xu), 0), xu])
                xt.append(xu[-window:].T)
                yt.append(min(true_rul[u - 1], RUL_CAP))
            c = {"x_tr": x_tr[sel], "y_tr": y_tr[sel], "x_va": x_va, "y_va": y_va,
                 "x_te": np.stack(xt), "y_te": np.array(yt, np.float32),
                 "meta": {"client": f"FD00{fd}_g{g}", "fd": fd, "train_units": tr_units.tolist(),
                          "val_units": va_units.tolist(), "test_units": t_units.tolist()}}
            clients.append(c)
    return clients


LOADERS = {"cwru": cwru_clients, "paderborn": paderborn_clients, "mimii": mimii_clients, "cmapss": cmapss_clients}
TASK = {"cwru": "cls", "paderborn": "cls", "mimii": "ae", "cmapss": "rul", "dirichlet": "cls"}
N_OUT = {"cwru": 4, "paderborn": 3, "mimii": 0, "cmapss": 1, "dirichlet": 4}


def load(dataset, **kw):
    if dataset == "dirichlet":
        return dirichlet_clients(**kw)
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{dataset}_clients.npz")
    if dataset != "mimii" and os.path.exists(path):
        return list(np.load(path, allow_pickle=True)["clients"])
    cl = LOADERS[dataset]()
    if dataset != "mimii":
        np.savez(path, clients=np.array(cl, dtype=object))
    return cl


def manifest(dataset, clients):
    """Partition manifest (sizes, class counts, recording/engine assignment) for the supporting files."""
    rows = []
    for c in clients:
        r = dict(c["meta"])
        for s in ("tr", "va", "te"):
            y = c[f"y_{s}"]
            r[f"n_{s}"] = int(len(y))
            if TASK.get(dataset) != "rul":
                r[f"classes_{s}"] = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        rows.append(r)
    return rows


if __name__ == "__main__":
    import sys
    for ds in sys.argv[1:]:
        cl = load(ds)
        print(ds, len(cl), "clients")
        os.makedirs(os.path.join(ROOT, "manifests"), exist_ok=True)
        with open(os.path.join(ROOT, "manifests", f"{ds}_partition.json"), "w") as f:
            json.dump(manifest(ds, cl), f, indent=1)
        for c in cl[:3]:
            print(" ", c["meta"]["client"], {k: c[k].shape for k in c if k.startswith("x_")})
