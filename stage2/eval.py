import logging
import os

import hydra
from pytorch_lightning import Trainer, seed_everything
import torch

from stage2.combined_learner import CombinedLearner
from stage2.data.combined_dm import DataModule

# static vars
logging.getLogger("lightning").propagate = False
__spec__ = None


@hydra.main(config_path="conf", config_name="config_combined")
def main(cfg):
    cfg.gpus = torch.cuda.device_count()
    if cfg.gpus < 2:
        cfg.trainer.accelerator = None

    learner = CombinedLearner(cfg)
    data_module = DataModule(cfg, root=os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    if cfg.model.weights_filename:
        df_weights_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "weights", cfg.model.weights_filename
        )
        state_dict = torch.load(df_weights_path)
        weights_backbone = {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("backbone")}
        learner.model.backbone.load_state_dict(weights_backbone)
        weights_df_head = {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("df_head")}
        learner.model.df_head.load_state_dict(weights_df_head)
        print("Weights loaded.")

    # Train
    trainer = Trainer(**cfg.trainer)
    trainer.test(learner, datamodule=data_module)


if __name__ == "__main__":
    seed_everything(42)
    main()
