import torch
import torch.nn.functional as F


roi = {
    "left": ("FC3", "C3", "CP3"),
    "right": ("FC4", "C4", "CP4"),
    "mid": ("FCz", "Cz", "CPz"),
}


def roi_index(names, spec=roi):
    z = {k: i for i, k in enumerate(names)}
    out = {}
    for k, v in spec.items():
        miss = [u for u in v if u not in z]
        if miss:
            raise ValueError(f"missing roi channels: {miss}")
        out[k] = [z[u] for u in v]
    return out


def roi_embed(x, names, spec=roi):
    if x.ndim != 3:
        raise ValueError(f"expected [b,c,t], got {tuple(x.shape)}")
    q = roi_index(names, spec)
    l = x[:, q["left"], :].mean(1)
    r = x[:, q["right"], :].mean(1)
    m = x[:, q["mid"], :].mean(1)
    a = (l - r).abs()
    v = torch.cat((l, r, a, m), 1)
    v = F.normalize(v, p=2, dim=1)
    return {"left": l, "right": r, "mid": m, "asym": a, "roi": v}


def roi_bank(v, y, n_cls=2, floor=0.7):
    if v.ndim != 2:
        raise ValueError(f"expected [b,d], got {tuple(v.shape)}")
    if y.ndim != 1:
        raise ValueError(f"expected [b], got {tuple(y.shape)}")
    v = F.normalize(v, p=2, dim=1)
    t, s = [], []
    for k in range(n_cls):
        g = v[y == k]
        if g.numel() == 0:
            raise ValueError(f"empty class: {k}")
        c = F.normalize(g.mean(0, keepdim=True), p=2, dim=1)[0]
        u = g @ c
        t.append(c)
        s.append(max(float(u.mean().item() - u.std().item()), float(floor)))
    return torch.stack(t, 0), torch.tensor(s, dtype=v.dtype, device=v.device)

class PGTC:
    def __init__(self, tau=0.6):
        self.tau = tau

    def __call__(self, p, r, t, d, h=None):
        if p.ndim != 2:
            raise ValueError(f"expected [b,c], got {tuple(p.shape)}")
        if r.ndim != 2:
            raise ValueError(f"expected [b,d], got {tuple(r.shape)}")
        r = F.normalize(r, p=2, dim=1)
        t = F.normalize(t, p=2, dim=1)
        c, y = p.max(1)
        s = (r * t[y]).sum(1)
        m = (c >= self.tau) & (s >= d[y])
        z = torch.zeros_like(p)
        z[m, y[m]] = 1.0
        q = z if h is None else h.clone()
        q[m] = z[m]
        return {"pseudo": q, "mask": m, "label": y, "conf": c, "sim": s}

