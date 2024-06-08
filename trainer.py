import os
import torch


class Trainer:
    def __init__(
            self, model: torch.nn.Module,
            data: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            config
    ):
        # 模型，数据，算法, 训练配置
        self.model = model.to(config.device)
        self.data = data
        self.optimizer = optimizer
        self.loss_fn = config.loss_fn
        self.config = config

        self.loss_list = list()
        self.start_epoch = 0
        self.epoch = 0
        self.num_epochs = config.num_epochs
        self.snapshot_path = config.snapshot_path
        self.device = config.device
        self.save_interval = config.save_interval
        if config.snapshot_path is not None and os.path.exists(path=config.snapshot_path):
            self.load_from_snapshot(config.snapshot_path)

    def load_from_snapshot(self, path):
        self.optimizer: torch.nn.Module
        self.model: torch.nn.Module
        snapshot = torch.load(path, map_location=self.device)
        self.model.load_state_dict(snapshot["model_parameters"])
        self.optimizer.load_state_dict(snapshot["optimizer_state"])
        self.epoch = snapshot["epoch"]
        self.start_epoch = self.epoch

    def save(self, path):
        self.optimizer: torch.nn.Module
        self.model: torch.nn.Module
        snapshot = dict()
        snapshot["model_parameters"] = self.model.state_dict()
        snapshot["optimizer_state"] = self.optimizer.state_dict()
        snapshot["epoch"] = self.epoch
        torch.save(snapshot, path)

    def train_a_batch(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        o = self.model(x)
        loss = self.loss_fn(o, y)
        self.loss_list.append(loss.item())
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def train_a_epoch(self):
        for x, y in self.data:
            self.train_a_batch(x, y)

    def train(self):
        for i in range(self.start_epoch, self.config.num_epochs):
            self.epoch = i
            self.train_a_epoch()
            if self.snapshot_path is not None and i % self.save_interval == 0:
                self.save(self.snapshot_path)

