import logging
import math
import os

import hydra
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
import torch

from stage2.combined_learner import CombinedLearner
from stage2.data.combined_dm import DataModule

# static vars
os.environ["WANDB_SILENT"] = "true"
logging.getLogger("lightning").propagate = False
# __spec__ = None


@hydra.main(config_path="conf", config_name="config_combined")
def main(cfg):
    cfg.gpus = torch.cuda.device_count()
    if cfg.gpus < 2:
        cfg.trainer.accelerator = None

    learner = CombinedLearner(cfg)
    data_module = DataModule(cfg, root=os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    wandb_logger = instantiate(cfg.logger)
    callbacks = [
        instantiate(cfg.checkpoint),
        LearningRateMonitor(logging_interval=cfg.logging.logging_interval),
    ]

    if cfg.model.weights_filename:
        weights_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "weights", cfg.model.weights_filename
        )
        state_dict = torch.load(weights_path)
        weights_student = {".".join(k.split(".")[3:]): v for k, v in state_dict.items() if
                           k.startswith("model.model1.backbone")}  # just for backbone
        learner.model.backbone.load_state_dict(weights_student, strict=True)

        if not math.isclose(cfg.model.ssl_weight, 0.0):
            weights_teacher = {".".join(k.split(".")[3:]): v for k, v in state_dict.items() if
                                k.startswith("model.model1.target_encoder")}  # just for target encoder
            learner.model.target_encoder.load_state_dict(weights_teacher, strict=True)

    # Train
    trainer = Trainer(
        **cfg.trainer,
        logger=wandb_logger,
        callbacks=callbacks,
    )
    trainer.fit(learner, data_module)


if __name__ == "__main__":
    seed_everything(42)
    main()
