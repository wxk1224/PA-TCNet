import math

import torch
import torch.nn.functional as F
from torch import nn


class _S(nn.Module):
    def __init__(self, c0, c1, k, p):
        super().__init__()
        self.a = nn.Conv2d(c0, c0, kernel_size=k, groups=c0, padding=p, bias=False)
        self.b = nn.Conv2d(c0, c1, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        return self.b(self.a(x))


class LocalMotorEncoder(nn.Module):
    def __init__(
        self,
        channels,
        width=30,
        seed=10,
        fold=3,
        pool=(8, 8),
        drop=0.35,
        span=(36, 24, 18),
        fuse=16,
    ):
        super().__init__()
        self.q = nn.ModuleList([nn.Conv2d(1, seed, (1, k), padding="same", bias=False) for k in span])
        c = seed * len(span)
        h = c * fold
        self.w = nn.BatchNorm2d(c)
        self.e = nn.Conv2d(c, h, (channels, 1), groups=c, bias=False)
        self.r = nn.BatchNorm2d(h)
        self.t = nn.MaxPool2d((1, pool[0]), (1, pool[0]))
        self.y = nn.Dropout(drop)
        self.u = _S(h, h, (1, fuse), "same")
        self.i = nn.BatchNorm2d(h)
        self.o = nn.MaxPool2d((1, pool[1]), (1, pool[1]))
        self.p = nn.Dropout(drop)
        self.s = nn.Identity() if h == width else nn.Linear(h, width)

    def forward(self, x):
        z = torch.cat([m(x) for m in self.q], 1)
        z = F.elu(self.w(z))
        z = F.elu(self.r(self.e(z)))
        z = self.y(self.t(z))
        z = F.elu(self.i(self.u(z)))
        z = self.p(self.o(z))
        z = z.flatten(2).transpose(1, 2)
        return self.s(z)


class _P(nn.Module):
    def __init__(self, d, n, drop=0.1):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, n, d))
        self.b = nn.Dropout(drop)

    def forward(self, x):
        return self.b(x + self.a[:, : x.size(1)].to(x.device))


def _avg(z):
    k = z.kernel_size[0]
    with torch.no_grad():
        z.weight.zero_()
        z.weight[:] = 1.0 / k


class _D(nn.Module):
    def __init__(self, d, k=9):
        super().__init__()
        self.a = nn.Conv1d(d, d, kernel_size=k, padding=k // 2, groups=d, bias=False)
        _avg(self.a)

    def forward(self, x):
        v = x.transpose(1, 2)
        l = self.a(v)
        return l, v - l


class _F(nn.Module):
    def __init__(self, d, lk=(5, 9, 13), hk=(3, 5, 7), mode="full"):
        super().__init__()
        self.m = mode
        self.a = _D(d, 9)
        self.b = nn.ModuleList([nn.Conv1d(d, d, k, padding=k // 2, groups=d, bias=False) for k in lk])
        self.c = nn.ModuleList([nn.Conv1d(d, d, k, padding=k // 2, groups=d, bias=False) for k in hk])
        self.d = nn.Sequential(nn.Conv1d(d * len(lk), d, 1, bias=False), nn.BatchNorm1d(d), nn.SiLU())
        self.e = nn.Sequential(nn.Conv1d(d * len(hk), d, 1, bias=False), nn.BatchNorm1d(d), nn.SiLU())
        self.f = nn.Sequential(
            nn.Conv1d(d * 2, d * 2, 1, bias=False),
            nn.BatchNorm1d(d * 2),
            nn.SiLU(),
            nn.Conv1d(d * 2, d, 1, bias=False),
        )
        self.g = nn.LayerNorm(d)

    def forward(self, x):
        l0, h0 = self.a(x)
        l1 = self.d(torch.cat([u(l0) for u in self.b], 1)) + l0
        h1 = self.e(torch.cat([u(h0) for u in self.c], 1)) + h0
        if self.m == "low_only":
            la, ha, rr = l1, torch.zeros_like(h1), l1
        elif self.m == "high_only":
            la, ha, rr = torch.zeros_like(l1), h1, h1
        else:
            la, ha, rr = l1, h1, 0.5 * (l1 + h1)
        c = self.f(torch.cat((la, ha), 1)) + rr
        return l1.transpose(1, 2), h1.transpose(1, 2), self.g(c.transpose(1, 2))


class _G(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.a = nn.LayerNorm(d)
        self.b = nn.Linear(d, h)
        self.c = nn.Linear(d, h)

    def forward(self, x):
        x = self.a(x)
        return torch.sigmoid(self.b(x)), self.c(x)


class _X(nn.Module):
    def __init__(self, d, mode="full"):
        super().__init__()
        self.d = d
        self.h = d * 2
        self.r = math.ceil(d / 16)
        self.s = 16
        self.m = mode
        self.a = nn.Linear(d, self.h * 2)
        self.b = nn.Conv1d(self.h, self.h, 3, groups=self.h, padding=2)
        self.c = nn.Linear(self.h, self.r + self.s * 2, bias=False)
        self.e = nn.Linear(self.r, self.h, bias=True)
        z = torch.arange(1, self.s + 1).repeat(self.h, 1)
        self.f = nn.Parameter(torch.log(z.float()))
        self.g = nn.Parameter(torch.ones(self.h))
        self.i = nn.Linear(self.h, d)
        self.o = _G(d, self.h) if mode != "none" else None

    def _scan(self, u, dt, a, b, c, d):
        B, L, H = u.shape
        S = a.shape[1]
        aa = torch.exp(torch.einsum("bld,dn->bldn", dt, a))
        bb = torch.einsum("bld,bln,bld->bldn", dt, b, u)
        x = torch.zeros(B, H, S, device=u.device, dtype=u.dtype)
        y = []
        for j in range(L):
            x = aa[:, j] * x + bb[:, j]
            y.append(torch.einsum("bdn,bn->bd", x, c[:, j]))
        y = torch.stack(y, 1)
        return y + u * d

    def _ssm(self, x):
        n = self.f.shape[1]
        a = -torch.exp(self.f.float())
        d = self.g.float()
        z = self.c(x)
        dt, b, c = z.split([self.r, n, n], -1)
        dt = F.softplus(self.e(dt))
        return self._scan(x, dt, a, b, c, d)

    def forward(self, x, ctx=None):
        z = self.a(x)
        u, v = z.split([self.h, self.h], -1)
        if self.o is not None and ctx is not None:
            p, q = self.o(ctx)
            u = u * (1.0 + p)
            v = v + q
        u = self.b(u.transpose(1, 2))[:, :, : x.size(1)].transpose(1, 2)
        u = F.silu(u)
        y = self._ssm(u)
        y = y * F.silu(v)
        return self.i(y)


class _B(nn.Module):
    def __init__(self, d, drop=0.35, mode="full"):
        super().__init__()
        self.a = nn.LayerNorm(d)
        self.b = _F(d, mode=mode) if mode != "none" else None
        self.c = _X(d, mode=mode)
        self.d = nn.Dropout(drop)

    def forward(self, x, trace=False):
        z = self.a(x)
        l = h = c = None
        if self.b is not None:
            l, h, c = self.b(z)
        y = x + self.d(self.c(z, c))
        if not trace:
            return y
        return y, {"low": l, "high": h, "context": c}


class PRSM(nn.Module):
    def __init__(self, width=30, depth=2, drop=0.35, mode="full"):
        super().__init__()
        self.a = nn.ModuleList([_B(width, drop=drop, mode=mode) for _ in range(depth)])

    def forward(self, x, trace=False):
        q = None
        for m in self.a:
            if trace:
                x, q = m(x, True)
            else:
                x = m(x, False)
        if not trace:
            return x
        return x, q


class PATCNet(nn.Module):
    def __init__(
        self,
        channels,
        samples,
        n_classes=2,
        width=30,
        depth=2,
        seed=10,
        fold=3,
        pool=(8, 8),
        drop=0.35,
        span=(36, 24, 18),
        fuse=16,
        mode="full",
    ):
        super().__init__()
        self.a = LocalMotorEncoder(
            channels=channels,
            width=width,
            seed=seed,
            fold=fold,
            pool=pool,
            drop=drop,
            span=span,
            fuse=fuse,
        )
        with torch.no_grad():
            n = self.a(torch.zeros(1, 1, channels, samples)).size(1)
        self.b = _P(width, n, 0.1)
        self.c = PRSM(width=width, depth=depth, drop=drop, mode=mode)
        self.d = nn.LayerNorm(width)
        self.e = nn.Sequential(nn.Dropout(0.5), nn.Linear(n * width, n_classes))

    def forward(self, x, trace=False):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        z = self.b(self.a(x) * math.sqrt(self.d.normalized_shape[0]))
        if trace:
            h, u = self.c(z, True)
        else:
            h = self.c(z, False)
            u = None
        y = self.e(self.d(h).flatten(1))
        if not trace:
            return y
        out = {"logits": y, "tokens": z}
        if u is not None:
            out.update(u)
        return out
