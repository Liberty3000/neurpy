import torch as th
from .attention import PatchAttention

class PreNorm(th.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn, self.norm = fn, th.nn.LayerNorm(dim)
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(th.nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.layers = th.nn.Sequential(
        th.nn.Linear(dim, hidden_dim),
        th.nn.GELU(),
        th.nn.Dropout(dropout),
        th.nn.Linear(hidden_dim, dim),
        th.nn.Dropout(dropout))
    def forward(self, x):
        return self.layers(x)

class Transformer(th.nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = th.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(th.nn.ModuleList([
            PreNorm(dim, PatchAttention(dim, heads=heads, head_dim=head_dim, dropout=dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
