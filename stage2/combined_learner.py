import math

from hydra.utils import instantiate
from pytorch_lightning.core.lightning import LightningModule

from stage2.metrics import VideoLevelAUROC, VideoLevelAcc, VideoLevelAUROCCDF, VideoLevelAUROCDFDC
from stage2.schedulers.warmup_cosine import WarmupCosineScheduler


class CombinedLearner(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = instantiate(cfg.model.obj, cfg)
        types = cfg.data.dataset_df.types_val
        if cfg.data.dataset_df.only_ff_val:
            types.remove("DeeperForensics")
            types.remove("FaceShifter")
        if len(types) > 0:
            self.auroc_ff = VideoLevelAUROC(types)
            self.acc_ff = VideoLevelAcc(types)
        self.auroc_cdf = VideoLevelAUROCCDF(("Real", "Fake"), multi_gpu=cfg.gpus > 1)
        self.auroc_dfdc = VideoLevelAUROCDFDC(multi_gpu=cfg.gpus > 1)
        if cfg.debug.log_gradients:
            self.logger.experiment.watch(self.model, log="gradients")

    def forward(self, videos_df, labels_df, videos_df_clean, videos_ssl, videos_ssl_clean):
        return self.model(videos_df, labels_df, videos_df_clean, videos_ssl, videos_ssl_clean)

    def training_step(self, data, batch_idx):
        videos_ssl = videos_ssl_clean = None
        if self.cfg.only_df:
            data_df = data[0]
        else:
            data_ssl, data_df = data
            videos_ssl, videos_ssl_clean = data_ssl["video_aug"], data_ssl["video"]

        videos_df, videos_df_clean, labels_df = data_df["video_aug"], data_df["video"], data_df["label"]

        loss_df, loss_ssl = self.forward(videos_df, labels_df, videos_df_clean, videos_ssl, videos_ssl_clean)

        self.log("loss_df", loss_df, on_step=True, prog_bar=True, on_epoch=True)
        if loss_ssl is not None:
            self.log("loss_ssl", loss_ssl, on_step=True, prog_bar=True, on_epoch=True)
        return self.cfg.model.ssl_weight * loss_ssl + loss_df

    def validation_step_df(self, data, ds_type, metric_auc, metric_acc=None):
        videos, labels, video_idxs = data["video"], data["label"], data["video_index"]
        logits = self.model.df_head(self.model.backbone(videos))

        if ds_type:
            metric_auc.update(logits, labels, video_idxs, ds_type)
        else:
            metric_auc.update(logits, labels, video_idxs)
        if metric_acc is not None:
            metric_acc.update(logits, labels, video_idxs, ds_type)

    def validation_step(self, data, batch_idx, dataloader_idx):
        if self.cfg.data.dataset_df.aggregate_scores:
            ds_type = "Real" if self.cfg.data.dataset_df.types_val[dataloader_idx] == "Real" else "FaceForensics"
        else:
            ds_type = self.cfg.data.dataset_df.types_val[dataloader_idx]
        self.validation_step_df(data, ds_type, self.auroc_ff, self.acc_ff)

    def validation_epoch_end(self, outputs):
        auroc_ff = self.auroc_ff.compute()
        self.log_dict(auroc_ff)
        self.auroc_ff.reset()

        acc_res = self.acc_ff.compute()
        self.log_dict(acc_res)
        self.acc_ff.reset()

    def test_step(self, data, batch_idx, dataloader_idx):
        num_ff_types = len(self.cfg.data.dataset_df.types_val)
        if dataloader_idx < num_ff_types:
            if self.cfg.data.dataset_df.aggregate_scores:
                ds_type = "Real" if self.cfg.data.dataset_df.types_val[dataloader_idx] == "Real" else "FaceForensics"
            else:
                ds_type = self.cfg.data.dataset_df.types_val[dataloader_idx]
            self.validation_step_df(data, ds_type, self.auroc_ff, self.acc_ff)
        elif dataloader_idx in (num_ff_types, num_ff_types + 1):
            ds_type = "Real" if dataloader_idx == num_ff_types else "Fake"
            self.validation_step_df(data, ds_type, self.auroc_cdf)
        else:
            self.validation_step_df(data, None, self.auroc_dfdc)

    def test_epoch_end(self, outputs):
        if len(self.cfg.data.dataset_df.types_val) > 0:
            self.log_dict(self.auroc_ff.compute())
            self.auroc_ff.reset()

            self.log_dict(self.acc_ff.compute())
            self.acc_ff.reset()

        if self.cfg.data.dataset_df.cdf_dfdc_test:
            self.log_dict(self.auroc_cdf.compute())
            self.auroc_cdf.reset()

            self.log_dict(self.auroc_dfdc.compute())
            self.auroc_dfdc.reset()

    def configure_optimizers(self):
        scale_factor = self.cfg.batch_size / 256
        if self.cfg.optimizer.optim.scale_sqrt:  # sqrt scaling for adaptive optimisers
            scale_factor = math.sqrt(scale_factor)
        lr_video = self.cfg.optimizer.base_lr_video * scale_factor  # linear scaling rule
        params = list(self.model.parameters())
        optimizer_video = instantiate(self.cfg.optimizer.optim.obj, params, lr=lr_video)

        train_len = self.cfg.data.dataset_df.videos_per_type * (len(self.cfg.data.dataset_df.fake_types_train) + 1)
        iter_per_epoch = train_len / (self.cfg.batch_size * self.cfg.trainer.accumulate_grad_batches)
        scheduler = WarmupCosineScheduler(
            optimizer_video,
            lr_video,
            self.cfg.optimizer.warmup_epochs,
            self.cfg.trainer.max_epochs,
            iter_per_epoch,
            self.cfg.optimizer.cosine_decay,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer_video], [scheduler]
