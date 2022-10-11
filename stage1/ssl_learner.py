import math

from hydra.utils import instantiate
import torch
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import Accuracy

from stage1.schedulers.warmup_cosine import WarmupCosineScheduler


@torch.no_grad()
def compute_std(features):
    return torch.std(F.normalize(features, dim=1), dim=0).mean()


class SSLLearner(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model = instantiate(cfg.model.obj, cfg)

        self.prober_video = instantiate(cfg.prober, in_dim=cfg.model.visual_backbone.output_dim)
        self.prober_audio = instantiate(cfg.prober, in_dim=cfg.model.audio_backbone.output_dim)

        self.acc_train_v = Accuracy()
        self.acc_train_a = Accuracy()
        self.acc_val_v = Accuracy()
        self.acc_val_a = Accuracy()

        if cfg.debug.log_gradients:
            self.logger.experiment.watch(self.model, log="gradients")
        self.automatic_optimization = False

    def forward(self, video, audio, mask, length, mode):
        return self.model(video, audio, mask, length, mode)

    def training_step(self, data, batch_idx):
        opt_video, opt_audio = self.optimizers()
        sch_video, sch_audio = self.lr_schedulers()

        # update momentum encoders
        self.model.update_moving_average_video()
        self.model.update_moving_average_audio()

        audio_targets = data["audio"] if self.cfg.data.clean_targets else data["audio_aug"]
        duration = [self.cfg.data.num_frames] * data["video"].size(0)
        loss_v2a, embeddings_video = self.forward(data["video_aug"], audio_targets, data["mask"], duration, 0)

        logits_v = self.prober_video(embeddings_video.detach())
        preds_v = torch.argmax(logits_v, dim=-1)
        acc_v = self.acc_train_v(preds_v, data["label"])
        loss_v = F.cross_entropy(logits_v, data["label"])
        self.log("acc_train_video", acc_v, on_step=False, on_epoch=True)

        opt_video.zero_grad()
        self.manual_backward(loss_v2a + loss_v)
        opt_video.step()
        sch_video.step()

        self.log("loss_v2a", loss_v2a, on_step=False, on_epoch=True, prog_bar=True)
        self.log("std_video", compute_std(embeddings_video), on_step=False, on_epoch=True, prog_bar=True)

        video_targets = data["video"] if self.cfg.data.clean_targets else data["video_aug"]
        loss_a2v, embeddings_audio = self.forward(video_targets, data["audio_aug"], data["mask"], duration, 1)

        logits_a = self.prober_audio(embeddings_audio.detach())
        preds_a = torch.argmax(logits_a, dim=-1)
        acc_a = self.acc_train_a(preds_a, data["label"])
        loss_a = F.cross_entropy(logits_a, data["label"])
        self.log("acc_train_audio", acc_a, on_step=False, on_epoch=True)

        opt_audio.zero_grad()
        self.manual_backward(loss_a2v + loss_a)
        opt_audio.step()
        sch_audio.step()

        self.log("loss_a2v", loss_a2v, on_step=False, on_epoch=True, prog_bar=True)
        self.log("std_audio", compute_std(embeddings_audio), on_step=False, on_epoch=True, prog_bar=True)

    def shared_val_test_step(self, data):
        videos, mels, labels = data["video"], data["audio"], data["label"]
        duration = [self.cfg.data.num_frames] * data["video"].size(0)
        mask = torch.ones((videos.size(0), self.cfg.data.num_frames), dtype=torch.long, device=videos.device)

        loss_v2a, embeddings_video = self.forward(videos, mels, mask, duration, 0)
        loss_a2v, embeddings_audio = self.forward(videos, mels, mask, duration, 1)

        self.log("loss_v2a_val", loss_v2a, on_step=False, on_epoch=True)
        self.log("loss_a2v_val", loss_a2v, on_step=False, on_epoch=True)

        logits_v, logits_a = self.prober_video(embeddings_video), self.prober_audio(embeddings_audio)
        preds_v, preds_a = torch.argmax(logits_v, dim=-1), torch.argmax(logits_a, dim=-1)
        self.acc_val_v.update(preds_v, labels), self.acc_val_a.update(preds_a, labels)

    def validation_step(self, data, batch_idx):
        self.shared_val_test_step(data)

    def test_step(self, data, batch_idx):
        self.shared_val_test_step(data)

    def validation_epoch_end(self, outputs):
        self.log_dict(
            {"acc_val_video_epoch": self.acc_val_v.compute(), "acc_val_audio_epoch": self.acc_val_a.compute()}
        )
        self.acc_val_v.reset(), self.acc_val_a.reset()

    def test_epoch_end(self, outputs):
        print({"acc_val_video_epoch": self.acc_val_v.compute(), "acc_val_audio_epoch": self.acc_val_a.compute()})
        self.acc_val_v.reset(), self.acc_val_a.reset()

    # potentially want different schedulers for predictors and rest of model
    def configure_optimizers(self):
        def get_param_groups(model, lr, incl_encoder=True, incl_predictor=True, prober=None, lr_prober=None):
            param_groups = []
            if incl_encoder:
                param_groups.append(
                    {
                        "name": "encoder",
                        "params": [
                            param for name, param in model.named_parameters() if not name.startswith("predictor")
                        ],
                        "lr": lr,
                    }
                )
            if incl_predictor:
                param_groups.append(
                    {
                        "name": "predictor",
                        "params": [param for name, param in model.named_parameters() if name.startswith("predictor")],
                        "lr": lr,
                    }
                )
            if prober is not None:
                param_groups.append({"name": "prober", "params": list(prober.parameters()), "lr": lr_prober})
            return param_groups

        scale_factor = self.cfg.batch_size / 256
        scale_factor_prober = self.cfg.batch_size_prober / 256
        if self.cfg.optimizer.optim.scale_sqrt:  # sqrt scaling for adaptive optimisers
            scale_factor = math.sqrt(scale_factor)
            scale_factor_prober = math.sqrt(scale_factor_prober)
        lr_video = self.cfg.optimizer.base_lr_video * scale_factor  # linear scaling rule
        lr_audio = self.cfg.optimizer.base_lr_audio * scale_factor
        lr_prober = self.cfg.optimizer.base_lr_prober * scale_factor_prober

        param_groups_video = get_param_groups(
            self.model.model1,
            lr_video,
            incl_encoder=True,
            prober=self.prober_video,
            lr_prober=lr_prober
        )
        param_groups_audio = get_param_groups(
            self.model.model2,
            lr_audio,
            incl_encoder=True,
            prober=self.prober_audio,
            lr_prober=lr_prober
        )

        optimizer_video = instantiate(self.cfg.optimizer.optim.obj, param_groups_video)
        optimizer_audio = instantiate(self.cfg.optimizer.optim.obj, param_groups_audio)

        if self.cfg.data.dataset.multiple_trainloader_mode == "max_size_cycle":
            train_len = max(
                self.cfg.data.dataset.train_len // self.cfg.batch_size, 488_763 // self.cfg.batch_size_prober
            )
        else:
            train_len = min(
                self.cfg.data.dataset.train_len // self.cfg.batch_size, 488_763 // self.cfg.batch_size_prober
            )

        warmup_epochs = self.cfg.optimizer.warmup_epochs
        scheduler_video = WarmupCosineScheduler(
            optimizer_video,
            warmup_epochs,
            self.cfg.trainer.max_epochs,
            train_len,
            self.cfg.optimizer.cosine_decay,
            excluded_groups=None if self.cfg.optimizer.schedule_predictors else ("predictor",)
        )
        scheduler_audio = WarmupCosineScheduler(
            optimizer_audio,
            warmup_epochs,
            self.cfg.trainer.max_epochs,
            train_len,
            self.cfg.optimizer.cosine_decay,
            excluded_groups=None if self.cfg.optimizer.schedule_predictors else ("predictor",)
        )

        return [optimizer_video, optimizer_audio], [scheduler_video, scheduler_audio]
