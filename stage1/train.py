import logging
import os

import hydra
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

from stage1.data.data_module import DataModule
from stage1.ssl_learner import SSLLearner

# static vars
os.environ["WANDB_SILENT"] = "true"
logging.getLogger("lightning").propagate = False
__spec__ = None


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    cfg.gpus = torch.cuda.device_count()
    if cfg.gpus < 2:
        cfg.trainer.accelerator = None

    wandb_logger = instantiate(cfg.logger)
    learner = SSLLearner(cfg)
    data_module = DataModule(cfg, root=os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    ckpt_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        dirpath=cfg.checkpoint.dirpath,
        save_last=True,
        filename=f'{cfg.experiment_name}-{{epoch}}'
    )
    callbacks = [
        ckpt_callback,
        LearningRateMonitor(logging_interval=cfg.logging.logging_interval),
    ]
    trainer = Trainer(**cfg.trainer, logger=wandb_logger, callbacks=callbacks)
    trainer.fit(learner, data_module)


if __name__ == "__main__":
    seed_everything(42)
    main()
