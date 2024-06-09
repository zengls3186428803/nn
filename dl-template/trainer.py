import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config_class.algorithmConfig import AlgorithmConfig
from utils.decorater import wandb_loger
from config_class.trainConfig import TrainConfig


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            dataloaders: Tuple[DataLoader, DataLoader, DataLoader],
            config: TrainConfig,
            algorithm_config: AlgorithmConfig
    ):
        # 模型，数据，算法, 训练配置
        self.model = model.to(config.device)
        self.train_dataloader = dataloaders[0]
        self.evaluate_dataloader = dataloaders[1]
        self.test_data_loader = dataloaders[2]

        self.optimizer = algorithm_config.optimizers[0](model.parameters(), lr=algorithm_config.learning_rate)
        self.scheduler = algorithm_config.optimizers[1](optimizer=self.optimizer,
                                                        lr_lambda=lambda epoch: algorithm_config.learning_rate)
        self.loss_fn = algorithm_config.loss_fn()

        self.config = config
        self.loss_list = list()
        self.start_epoch = 0
        self.epoch = 0
        self.num_epochs = config.num_epochs
        self.snapshot_path = config.snapshot_path
        self.device = config.device
        self.save_interval = config.save_interval
        if config.snapshot_path is not None and os.path.exists(path=config.snapshot_path):
            print("load from checkpoint(snapshot)")
            self.load_from_snapshot(config.snapshot_path)

    def load_from_snapshot(self, path):
        self.optimizer: torch.nn.Module
        self.model: torch.nn.Module
        snapshot = torch.load(path, map_location=self.device)
        self.model.load_state_dict(snapshot["model_parameters"])
        self.optimizer.load_state_dict(snapshot["optimizer_state"])
        self.epoch = snapshot["epoch"]
        self.start_epoch = self.epoch + 1

    def save(self, path):
        self.optimizer: torch.nn.Module
        self.model: torch.nn.Module
        snapshot = dict()
        snapshot["model_parameters"] = self.model.state_dict()
        snapshot["optimizer_state"] = self.optimizer.state_dict()
        snapshot["epoch"] = self.epoch
        torch.save(snapshot, path)

    @wandb_loger(desc="")
    def train_a_batch(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        o = self.model(x)
        loss = self.loss_fn(o, y)
        self.loss_list.append(loss.item())
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        # self.evaluate(self.evaluate_dataloader, flag="train")
        loss: torch.Tensor
        loss_val = float(loss.item())
        return {
            "train_step_loss": loss_val
        }

    def train_a_epoch(self):
        total = len(self.train_dataloader)
        p_bar = tqdm(total=total, desc="step(iteration)([mini]-batch)")
        i = 0
        for x, y in self.train_dataloader:
            self.train_a_batch(x, y)
            if i % self.config.evaluate_interval_steps == 0:
                self.evaluate(
                    self.test_data_loader,
                    flag=f"per {self.config.evaluate_interval_steps} step : "
                )
            i += 1
            p_bar.update(1)

    def train(self):
        p_bar = tqdm(total=self.num_epochs, desc="epoch")
        for i in range(self.start_epoch, self.config.num_epochs):
            self.epoch = i
            self.train_a_epoch()
            if self.snapshot_path is not None and i % self.save_interval == 0:
                self.save(self.snapshot_path)
            if i % self.config.evaluate_interval_epochs == 0:
                self.evaluate(self.test_data_loader, flag=f"per {self.config.evaluate_interval_epochs} epoch : ")
            self.scheduler.step(self.epoch)
            p_bar.update(1)

    @wandb_loger(desc="")
    def evaluate(self, dataloader: DataLoader, flag="train:"):
        with torch.no_grad():
            correct_total = 0
            all_total = 0
            loss_total = 0
            n_batch = len(dataloader)
            p_bar = tqdm(total=n_batch, desc="evaluate")
            for x, y in dataloader:
                x = x.to(self.config.device)
                y = y.to(self.config.device)
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
                accurate, correct, batch_size = self.compute_accurate(logits, y)
                # print(flag + ":" + f"loss={loss.item()},acc={accurate},correct/batch_size={int(correct)}/{batch_size}")
                all_total += batch_size
                correct_total += correct
                loss_total += loss.item()
                p_bar.update(1)
            print(flag + ":" +
                  f"average_loss={loss_total / n_batch},average_acc={correct_total / all_total},correct_total/all_total={correct_total}/{all_total}")
            average_loss = loss_total / n_batch
            return {
                flag + "average_loss": average_loss,
                flag + "correct_total": correct_total,
                flag + "all_total": all_total,
            }

    def compute_accurate(self, logits: torch.Tensor, true_val: torch.Tensor):
        with torch.no_grad():
            correct = (logits.argmax(-1) == true_val).type(torch.float).sum().item()
            shape = true_val.shape
            total = 1
            for i in shape:
                total *= i
            acc = correct / total
        return acc, correct, total

    def compute_loss_no_grad(self, predict_val, true_val):
        with torch.no_grad():
            loss = self.loss_fn(predict_val, true_val)
        return loss
