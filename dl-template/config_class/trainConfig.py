import torch
from omegaconf import DictConfig, OmegaConf


class TrainConfig:
    def __init__(self, cfg: DictConfig):
        self.snapshot_path = OmegaConf.select(cfg, "train.snapshot_path", default=None)
        self.num_epochs = OmegaConf.select(cfg, "train.num_epochs", default=10)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_interval = OmegaConf.select(cfg, "train.save_interval", default=5)
        self.evaluate_interval_steps = OmegaConf.select(cfg, "train.evaluate_interval_steps", default=10)
        self.evaluate_interval_epochs = OmegaConf.select(cfg, "train.evaluate_interval_epochs", default=10)