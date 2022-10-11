# adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

from einops import rearrange, repeat
import torch
from torch import nn, einsum

from stage1.models.utils import Lambda


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        if isinstance(self.norm, nn.BatchNorm1d):
            self.norm = nn.Sequential(
                Lambda(lambda x: x.transpose(1, 2)), self.norm, Lambda(lambda x: x.transpose(1, 2))
            )
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim), norm_layer=norm_layer),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TransformerProber(nn.Module):
    def __init__(
        self,
        num_classes,
        in_dim,
        dim,
        num_frames,
        heads,
        mlp_dim,
        depth,
        dropout,
        norm="ln",
        pool="cls",
        dim_head=64,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert norm in ("ln", "bn")
        norm_layer = nn.LayerNorm if norm == "ln" else nn.BatchNorm1d
        self.projection_layer = nn.Linear(in_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, norm_layer)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(norm_layer(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        b, n, _ = x.shape

        x = self.projection_layer(x)
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class LinearProber(nn.Module):
    def __init__(self, in_dim, out_features):
        super().__init__()
        self.net = nn.Linear(in_dim, out_features)

    def forward(self, x):
        return self.net(x)
