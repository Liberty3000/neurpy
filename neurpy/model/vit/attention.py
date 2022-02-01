import einops, torch as th

class PatchAttention(th.nn.Module):
    def __init__(self, dim, heads=8, head_dim=64, dropout=0.):
        super().__init__()
        inner_dim = head_dim *  heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = th.nn.Softmax(dim = -1)
        self.to_qkv = th.nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = th.nn.Sequential(
        th.nn.Linear(inner_dim, dim),
        th.nn.Dropout(dropout)
        if project_out else th.nn.Identity())

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = th.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = th.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
