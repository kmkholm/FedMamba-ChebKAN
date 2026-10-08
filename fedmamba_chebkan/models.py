"""Mamba encoder (pure PyTorch selective scan), Chebyshev-KAN head, 1-D CNN backbone and task heads."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ Mamba
def selective_scan(u, delta, A, B, C, D, chunk=8):
    """h_t = exp(delta_t A) h_{t-1} + delta_t B_t u_t,  y_t = C_t . h_t + D u_t   (diagonal A, per channel).
    u, delta: (b, l, d); A: (d, n); B, C: (b, l, n). Exact chunked evaluation: inside a chunk of `chunk` steps,
    h_t = exp(S_t) * cumsum_s(exp(-S_s) dBu_s) with S the within-chunk cumulative log decay; the per-step log decay is
    floored at -6 (a per-step decay below e^-6 = 0.25 % is already a reset), so |S| <= 6*chunk = 48 keeps exp() far inside float32 range.
    Chunks are chained sequentially through the carried state."""
    b, l, d = u.shape
    n = A.shape[1]
    pad = (-l) % chunk
    if pad:
        z = lambda t: F.pad(t, (0, 0, 0, pad))
        u, delta, B, C = z(u), z(delta), z(B), z(C)
    L = u.shape[1]
    nc = L // chunk
    dA = (delta.unsqueeze(-1) * A).clamp(min=-6.0).view(b, nc, chunk, d, n)     # log decay per step (<= 0)
    dBu = ((delta * u).unsqueeze(-1) * B.unsqueeze(2)).view(b, nc, chunk, d, n)
    S = torch.cumsum(dA, dim=2)
    eS = torch.exp(S)
    h_local = eS * torch.cumsum(torch.exp(-S) * dBu, dim=2)
    # carry between chunks: only the chunk-end states are chained (small tensors), then added back in one pass
    ends, decays = h_local[:, :, -1].unbind(1), eS[:, :, -1].unbind(1)
    carry = [torch.zeros(b, d, n, device=u.device, dtype=u.dtype)]
    for c in range(nc - 1):
        carry.append(decays[c] * carry[-1] + ends[c])
    H = (h_local + eS * torch.stack(carry, 1).unsqueeze(2)).view(b, L, d, n)
    y = torch.einsum("bldn,bln->bld", H, C.view(b, L, n)) + D * u
    return y[:, :l]


class MambaBlock(nn.Module):
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2):
        super().__init__()
        di = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * di)
        self.conv = nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1)
        self.x_proj = nn.Linear(di, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, di)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(di, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d_model)
        with torch.no_grad():
            dt = torch.exp(torch.rand(di) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.d_state = d_state

    def forward(self, x):                                   # x: (b, l, d_model)
        r = x
        x = self.norm(x)
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2))
        p = self.x_proj(u)
        dt, B, C = torch.split(p, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        y = selective_scan(u, delta, A, B, C, self.D)
        return r + self.out_proj(y * F.silu(z))


class MambaEncoder(nn.Module):
    """Patch embedding (Conv1d stride p, GELU, BatchNorm, learnable positions) -> N Mamba blocks -> mean pool."""

    def __init__(self, c_in, length, d_model=64, n_layers=3, patch=8, d_state=16, d_conv=4, pool="mean", norm="bn"):
        super().__init__()
        self.pool = pool
        self.embed = nn.Conv1d(c_in, d_model, patch, stride=patch)
        # BatchNorm mixes the examples of a batch; under DP-SGD the per-example GroupNorm is used instead
        self.bn = nn.BatchNorm1d(d_model) if norm == "bn" else nn.GroupNorm(8, d_model)
        n_tok = length // patch
        self.pos = nn.Parameter(torch.zeros(1, n_tok, d_model))
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, d_conv) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.dim = d_model

    def forward(self, x):                                   # x: (b, c_in, L)
        h = self.bn(F.gelu(self.embed(x))).transpose(1, 2)
        h = h + self.pos[:, : h.shape[1]]
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        return h[:, -1] if self.pool == "last" else h.mean(1)     # 'last': causal state at the final step (RUL)


class CNN1D(nn.Module):
    """1-D CNN backbone used for the FedAvg reference (ablation row A)."""

    def __init__(self, c_in, length, d_model=64):
        super().__init__()
        ch = [c_in, 16, 32, 64, d_model]
        layers = []
        for i in range(4):
            layers += [nn.Conv1d(ch[i], ch[i + 1], 7, padding=3), nn.BatchNorm1d(ch[i + 1]), nn.ReLU()]
            if length // (2 ** (i + 1)) >= 2:
                layers.append(nn.MaxPool1d(2))
        self.net = nn.Sequential(*layers)
        self.dim = d_model

    def forward(self, x):
        return self.net(x).mean(-1)


# ------------------------------------------------------------------ Chebyshev-KAN
class ChebKANLayer(nn.Module):
    """phi_ij(t) = w_base_ij * SiLU(t) + sum_n c_ijn T_n(t),  t = tanh(input) in [-1, 1].
    `coef` (in, out, N+1) is the Chebyshev coefficient state aggregated by SpectralFedAvg (w_cheb absorbed)."""

    def __init__(self, d_in, d_out, degree=8):
        super().__init__()
        self.degree = degree
        # smoothness prior: coefficient amplitude decays as (1+n)^-2, so the tail-energy criterion is informative from round 1
        decay = (1.0 + torch.arange(degree + 1, dtype=torch.float32)) ** -2.0
        self.coef = nn.Parameter(torch.randn(d_in, d_out, degree + 1) * decay / (d_in * decay.pow(2).sum()) ** 0.5)
        self.base = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.kaiming_uniform_(self.base, a=math.sqrt(5))

    def basis(self, t):                                     # t: (b, d_in) -> (b, d_in, N+1)
        T = [torch.ones_like(t), t]
        for _ in range(2, self.degree + 1):
            T.append(2 * t * T[-1] - T[-2])
        return torch.stack(T[: self.degree + 1], -1)

    def forward(self, x):
        t = torch.tanh(x)
        return F.linear(F.silu(t), self.base) + torch.einsum("bin,ion->bo", self.basis(t), self.coef)


class Net(nn.Module):
    """backbone in {mamba, cnn}; head in {kan, linear}; task in {cls, rul, ae}."""

    def __init__(self, task, n_out, c_in, length, backbone="mamba", head="kan", degree=8, patch=8, kan_hidden=32, norm="bn",
                 d_model=64, n_layers=3):
        super().__init__()
        self.task = task
        self.enc = (MambaEncoder(c_in, length, d_model=d_model, n_layers=n_layers, patch=patch,
                                 pool="last" if task == "rul" else "mean", norm=norm)
                    if backbone == "mamba" else CNN1D(c_in, length))
        d = self.enc.dim
        if head == "kan":
            self.kan1 = ChebKANLayer(d, kan_hidden, degree)
            self.kan2 = ChebKANLayer(kan_hidden, n_out, degree) if task == "cls" else None
            hid = kan_hidden
        else:
            self.kan1, self.kan2 = None, None
            hid = d
        if task == "cls":
            self.out = None if head == "kan" else nn.Linear(d, n_out)
        elif task == "rul":
            self.out = nn.Linear(hid, 1)
        else:                                               # autoencoder: bottleneck -> decoder
            self.c_in, self.length = c_in, length
            self.dec_fc = nn.Linear(hid, 64 * 40)
            self.dec_up = nn.ConvTranspose1d(64, c_in, 8, stride=8)

    def forward(self, x):
        f = self.enc(x)
        if self.task == "cls":
            if self.kan1 is not None:
                return self.kan2(self.kan1(f))
            return self.out(f)
        z = self.kan1(f) if self.kan1 is not None else f
        if self.task == "rul":
            return self.out(z).squeeze(-1)
        h = F.gelu(self.dec_fc(z)).view(-1, 64, 40)
        return self.dec_up(h)[..., : self.length]

    def kan_layers(self):
        return [m for m in (self.kan1, self.kan2) if m is not None]
