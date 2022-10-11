import torch.nn as nn

from stage1.models.utils import Lambda


def make_mlp_layer(in_dim, out_dim=4096, relu=True, norm_layer=nn.BatchNorm1d):
    layers = [nn.Linear(in_dim, out_dim)]
    if norm_layer:
        layers.append(norm_layer(out_dim))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# MLP used in SimSiam for predictor (with the below defaults) and in BYOL for both projector and predictor
class MLPPredictor(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512):
        super().__init__()

        norm_layer = nn.BatchNorm1d
        self.layer1 = make_mlp_layer(in_dim, hidden_dim, norm_layer=norm_layer)
        self.layer2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        n = x.size(1)
        x = x.reshape((-1, x.size(-1)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape((-1, n, x.size(-1)))
        return x


class MLPProjectorBYOL(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=4096, out_dim=256):
        super().__init__()

        norm_layer = nn.BatchNorm1d
        self.layer1 = make_mlp_layer(in_dim, hidden_dim, norm_layer=norm_layer)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        n = x.size(-1)
        x = x.transpose(1, 2).contiguous().view(-1, x.size(1))
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, n, x.size(-1))  # keep this for transformer
        return x


# MLP used in SimSiam for projector
class MLPProjector(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=2048, last_bn=True):
        super().__init__()
        norm_layer = nn.BatchNorm1d
        last_norm_layer = None
        if last_bn:
            last_norm_layer = nn.BatchNorm1d
        self.net = nn.Sequential(
            make_mlp_layer(in_dim, hidden_dim, norm_layer=norm_layer),
            make_mlp_layer(hidden_dim, hidden_dim, norm_layer=norm_layer),
            make_mlp_layer(hidden_dim, out_dim, relu=False, norm_layer=last_norm_layer),
        )

    def forward(self, x):
        n = x.size(-1)
        x = x.transpose(1, 2).contiguous().view(-1, x.size(1))
        x = self.net(x)
        x = x.view(-1, n, x.size(-1))  # keep this for transformer
        return x


class LinearProjector(nn.Module):
    def __init__(self, in_dim=512, out_dim=2048):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)
        self.norm_layer = nn.BatchNorm1d(out_dim)

    def forward(self, x):  # NxCxT -> NxTxC
        n = x.size(-1)
        x = x.transpose(1, 2).contiguous().view(-1, x.size(1))
        x = self.net(x)
        x = self.norm_layer(x)
        x = x.view(-1, n, x.size(-1))  # keep this for transformer
        return x


class LinearProber(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            Lambda(lambda x: x.mean(-1)),
            nn.Linear(in_features, out_features),
        )

    def forward(self, x):
        return self.net(x)


def make_mlp_layer_global(in_dim, out_dim=4096, relu=True, norm_layer=nn.BatchNorm1d):
    layers = [nn.Linear(in_dim, out_dim), norm_layer(out_dim)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# MLP used in SimSiam for predictor (with the below defaults) and in BYOL for both projector and predictor
class MLPPredictorGlobal(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512):
        super().__init__()

        norm_layer = nn.BatchNorm1d
        self.layer1 = make_mlp_layer_global(in_dim, hidden_dim, norm_layer=norm_layer)
        self.layer2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# MLP used in SimSiam for projector
class MLPProjectorGlobal(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=2048):
        super().__init__()

        norm_layer = nn.BatchNorm1d
        self.layer1 = make_mlp_layer_global(in_dim, hidden_dim, norm_layer=norm_layer)
        self.layer2 = make_mlp_layer_global(hidden_dim, hidden_dim, norm_layer=norm_layer)
        self.layer3 = make_mlp_layer_global(hidden_dim, out_dim, relu=False, norm_layer=norm_layer)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
