import math

from hydra.utils import instantiate
import torch
import torch.nn as nn
import torch.nn.functional as F

from stage2.models.utils import set_requires_grad


class ModelCombined(nn.Module):
    def __init__(self, cfg=None, scale=64):
        super().__init__()

        self.backbone = instantiate(cfg.model.visual_backbone.obj)
        self.df_head = instantiate(cfg.model.df_predictor, cfg.model.visual_backbone.output_dim, scale=scale)
        self.scale = scale

        self.ssl_weight = cfg.model.ssl_weight
        if not math.isclose(self.ssl_weight, 0.0):
            target_backbone = instantiate(cfg.model.visual_backbone.obj)
            self.target_encoder = nn.Sequential(
                target_backbone, instantiate(cfg.model.projector, in_dim=cfg.model.visual_backbone.output_dim)
            )
            set_requires_grad(self.target_encoder, False)
            self.ssl_head = nn.Sequential(
                instantiate(cfg.model.projector, in_dim=cfg.model.visual_backbone.output_dim),
                instantiate(cfg.model.predictor, in_dim=cfg.model.projection_size),
            )

        num_fakes = len(cfg.data.dataset_df.fake_types_train)
        prior_fake = num_fakes / ((num_fakes + 1) * (cfg.model.relative_bs + 1))

        self.logit_adj = (
            torch.log(torch.tensor(prior_fake) / (1.0 - torch.tensor(prior_fake))) if cfg.model.logit_adj else 0.0
        )

    @torch.no_grad()
    def get_targets(self, x_ssl):
        return self.target_encoder(x_ssl)

    def forward(self, videos_df, labels_df, videos_df_clean=None, videos_ssl=None, videos_ssl_clean=None):
        videos, videos_clean = videos_df, videos_df_clean
        if videos_ssl is not None:  # Treat SSL videos as real
            videos, videos_clean = torch.cat([videos_df, videos_ssl]), torch.cat([videos_df_clean, videos_ssl_clean])
            zeros = torch.zeros(videos_ssl.size(0), dtype=labels_df.dtype, device=labels_df.device)
            labels_df = torch.cat([labels_df, zeros])

        # DeepFake detection loss
        features = self.backbone(videos)
        logits = self.df_head(features)
        loss_df = F.binary_cross_entropy_with_logits(logits.squeeze(-1) + self.logit_adj, labels_df.float())

        # SSL loss
        loss_ssl = 0.0
        if not math.isclose(self.ssl_weight, 0.0):
            targets = self.get_targets(videos_clean[~labels_df.bool()])  # Only for real videos
            predictions = self.ssl_head(features[~labels_df.bool()])
            loss_ssl = -F.cosine_similarity(predictions, targets, dim=-1).mean()

        return loss_df, loss_ssl
