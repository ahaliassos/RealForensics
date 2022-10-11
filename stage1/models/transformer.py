# adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

from einops import rearrange
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
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

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
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout), norm_layer=norm_layer),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        dim,
        out_dim,
        heads,
        mlp_dim,
        depth,
        dropout,
        norm="ln",
        final_bn=False,
        class_token=False,
        use_mlp_head=False,
        transpose=False,
        pool=None,
        dim_head=64,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert norm in ("ln", "bn")
        norm_layer = nn.LayerNorm if norm == "ln" else nn.BatchNorm1d
        self.projection_layer = nn.Linear(in_dim, dim)

        self.class_token = class_token
        self.cls_token = None
        if class_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, norm_layer)

        self.transpose = transpose
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = None
        self.use_mlp_head = use_mlp_head
        if use_mlp_head:
            self.mlp_head = nn.Sequential(
                Lambda(lambda x: x.transpose(1, 2)),
                norm_layer(dim),
                Lambda(lambda x: x.transpose(1, 2)),
                nn.Linear(dim, out_dim),
            )

        self.final_bn = None
        if final_bn:
            self.final_bn = nn.Sequential(
                Lambda(lambda x: x.transpose(1, 2)), nn.BatchNorm1d(out_dim), Lambda(lambda x: x.transpose(1, 2)),
            )

    def forward(self, x):
        if self.transpose:
            x = x.transpose(1, 2).contiguous()
        b, n, _ = x.shape

        x = self.projection_layer(x)
        x = self.dropout(x)

        x = self.transformer(x)

        if self.pool:
            x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)

        if self.use_mlp_head:
            x = self.mlp_head(x)

        if self.final_bn:
            x = self.final_bn(x)

        return x
