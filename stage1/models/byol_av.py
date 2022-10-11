import copy

from hydra.utils import instantiate
import torch
import torch.nn as nn
import torch.nn.functional as F

from stage1.models.utils import EMA, set_requires_grad


class BYOLSingle(nn.Module):
    def __init__(self, cfg, backbone=None, beta=0.999):
        super().__init__()
        self.backbone = instantiate(backbone.obj)
        self.projector = instantiate(cfg.projector, in_dim=backbone.output_dim)
        self.predictor = instantiate(cfg.predictor) if cfg.use_predictor else None
        self.target_backbone, self.target_projector = self.get_target_model(self.backbone), self.get_target_model(
            self.projector)
        self.ema = EMA(beta)
        self.use_shuffle_bn = cfg.use_shuffle_bn

        self.use_global = cfg.use_global

    def update_moving_average(self):
        self.ema.update_moving_average(self.target_backbone, self.backbone)
        self.ema.update_moving_average(self.target_projector, self.projector)

    def get_target_model(self, model):
        target_model = copy.deepcopy(model)
        set_requires_grad(target_model, False)
        return target_model

    @torch.no_grad()
    def get_targets(self, x):
        e = self.target_backbone(x)
        if self.use_global:
            e = e.mean(-1)
        return self.target_projector(e)

    def get_predictions(self, x):
        e_o = self.backbone(x)
        if self.use_global:
            e_o = e_o.mean(-1)
        z_o = self.projector(e_o)
        p_o = self.predictor(z_o)
        return e_o, p_o

    def forward(self, x, return_targets=False):
        if return_targets:
            return self.get_targets(x)  # create targets for other nets
        else:
            return self.get_predictions(x)


class BYOLAV(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        # linear scaling rule for momentum
        self.visual_beta = 1 - (1 - cfg.model.visual_beta_base) * cfg.batch_size / 96
        self.audio_beta = 1 - (1 - cfg.model.audio_beta_base) * cfg.batch_size / 96
        self.model1 = BYOLSingle(cfg.model, cfg.model.visual_backbone, self.visual_beta)
        self.model2 = BYOLSingle(cfg.model, cfg.model.audio_backbone, self.audio_beta)
        self.use_global = cfg.model.use_global

    def update_moving_average_video(self):
        self.model1.update_moving_average()

    def update_moving_average_audio(self):
        self.model2.update_moving_average()

    def forward(self, video, audio, mask, length, mode):  # mode=0 for all targets, mode=1,2,3 for models 1,2,3 resp.
        if mode == 0:
            e_vo, p_vo = self.model1(video)
            z_at = self.model2(audio, return_targets=True)

            if self.use_global:
                return torch.mean(-F.cosine_similarity(p_vo, z_at)), e_vo

            return torch.mean(
                torch.stack(
                    [-F.cosine_similarity(v[m.bool()], a[m.bool()], dim=-1).mean() for v, a, m in zip(p_vo, z_at, mask)]
                )
            ), e_vo
        if mode == 1:
            e_ao, p_ao = self.model2(audio)
            z_vt = self.model1(video, return_targets=True)

            if self.use_global:
                return torch.mean(-F.cosine_similarity(p_ao, z_vt)), e_ao

            return torch.mean(
                torch.stack(
                    [-F.cosine_similarity(v[m.bool()], a[m.bool()], dim=-1).mean() for v, a, m in zip(p_ao, z_vt, mask)]
                )
            ), e_ao
