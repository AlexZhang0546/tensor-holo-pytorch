"""
通用训练器基类，提供训练循环的公共逻辑：
  - 模型保存/恢复
  - 日志记录
  - 验证循环框架
具体阶段（stage1/stage2）通过继承该类并实现相应的训练步骤来使用。
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class BaseTrainer:
    def __init__(self, model: nn.Module, device: torch.device, ckpt_dir: str):
        self.model = model
        self.device = device
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        # 训练状态
        self.start_epoch = 0
        self.global_step = 0

    def save_checkpoint(self, state: Dict[str, Any], filename: str):
        torch.save(state, os.path.join(self.ckpt_dir, filename))
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filename: str) -> Dict[str, Any]:
        filepath = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(filepath):
            print(f"Loading checkpoint: {filepath}")
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.global_step = checkpoint.get('global_step', 0)
            return checkpoint
        else:
            raise FileNotFoundError(f"No checkpoint found at {filepath}")

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """打印指标到控制台（可扩展为 TensorBoard）。"""
        log_str = f"{prefix} Step {step}: " + ", ".join(
            [f"{k}: {v:.6f}" for k, v in metrics.items()]
        )
        print(log_str)

    def validate(self, val_loader, *args, **kwargs) -> Dict[str, float]:
        """子类应实现具体验证逻辑，返回指标字典。"""
        raise NotImplementedError

    def train(self, train_loader, val_loader, *args, **kwargs):
        """子类应实现完整训练循环。"""
        raise NotImplementedError