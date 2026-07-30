# src/data/dataset.py
"""
PyTorch 数据集类，用于直接读取原始 TFRecord 文件。
保持与原 TFRecordExtractorforTH._extract_fn 完全一致的解析逻辑，
并返回模型可直接使用的 RGBD（LDI）输入及目标全息图（amp_4, phs_4）。
同时提供复数形式的目标全息图 target_complex。

依赖：
    pip install tfrecord
"""

import math
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Optional
import os

from tfrecord import reader as tfrecord_reader
from tfrecord import example_pb2


class THDataset(Dataset):
    """
    全息图训练数据集，继承自 torch.utils.data.Dataset。

    功能：
        - 直接读取 .tfrecord 文件，不修改原始数据；
        - 按 labels 列表解析特征（与原 TFRecordExtractorforTH._extract_fn 等价）；
        - 支持多层 LDI 输入（通过 active_max_ldi_layer 控制）；
        - 返回 dict 包含 'rgbd' (拼接后的 LDI 输入)、'amp_4'、'phs_4' 和 'target_complex'，
          与原始 _preprocess_input 逻辑一致。
    """

    def __init__(
        self,
        tfrecord_path: str,
        dataset_params: Dict[str, Any],
        labels: List[str],
        active_max_ldi_layer: int = 0,
        load_to_memory: bool = True
    ):
        """
        Args:
            tfrecord_path: TFRecord 文件完整路径。
            dataset_params: 包含 'res_h', 'res_w', 'sample_count' 等必要字段。
            labels: 特征键的列表，如 ['amp_4', 'phs_4', 'img_0', 'depth_0', ...]。
            active_max_ldi_layer: 最大 LDI 层索引，默认 0（单层 RGBD）。
            load_to_memory: 若为 True，则一次性将全部样本读入内存（适合中小规模数据集，
                            避免生成外部索引文件）。若 False，此处暂不支持（保留扩展能力）。
        """
        super().__init__()
        self.tfrecord_path = tfrecord_path
        self.dataset_params = dataset_params
        self.labels = labels
        self.active_max_ldi_layer = active_max_ldi_layer
        self.res_h = dataset_params['res_h']
        self.res_w = dataset_params['res_w']
        self.sample_count = dataset_params.get('sample_count', 0)

        # 解析 TFRecord 并将所有样本缓存到内存（列表）
        self.samples: List[Dict[str, np.ndarray]] = []
        if load_to_memory:
            self._load_all_samples()

    def _load_all_samples(self):
        """一次性遍历 TFRecord 文件，将每个样本解析并存入 self.samples。"""
        if not os.path.exists(self.tfrecord_path):
            raise FileNotFoundError(f"TFRecord not found: {self.tfrecord_path}")

        # 使用 tfrecord 包的原始读取器（无 TensorFlow 依赖）
        reader = tfrecord_reader.tfrecord_iterator(self.tfrecord_path)
        count = 0
        for record_bytes in reader:
            sample = self._parse_example(record_bytes)
            self.samples.append(sample)
            count += 1
        # 如果提供了 sample_count，可以进行一致性校验
        if self.sample_count and count != self.sample_count:
            print(f"Warning: expected {self.sample_count} samples, but found {count} in {self.tfrecord_path}")

    def _parse_example(self, record_bytes: bytes) -> Dict[str, np.ndarray]:
        """
        解析单条 TFRecord 字节串，返回特征 dict，值均为 float32 的 numpy 数组，
        形状为 (C, H, W)，其中 depth 类的特征通道数 C=1，其余为 3。

        与原 TFRecordExtractorforTH._extract_fn 完全一致。
        """
        # 用 tfrecord 自带的 protobuf 定义解析 Example
        ex = example_pb2.Example()
        ex.ParseFromString(record_bytes)

        features = {}
        for label in self.labels:
            raw_feature = ex.features.feature[label].float_list.value
            # 确定通道数：depth 开头为单通道，否则 3 通道
            if label.startswith("depth"):
                num_channels = 1
            else:
                num_channels = 3
            # 转为 numpy 并 reshape 为 (C, H, W)
            arr = np.array(raw_feature, dtype=np.float32).reshape((num_channels, self.res_h, self.res_w))
            features[label] = arr
        return features

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回一个样本的 dict，包含：
            'rgbd':    Tensor (input_dim, H, W) - LDI 拼接结果，与原始 _preprocess_input 中 rgbd 一致
            'amp_4':   Tensor (3, H, W) - 目标振幅
            'phs_4':   Tensor (3, H, W) - 目标相位
            'target_complex': Tensor (3, H, W) - 复数目标全息图（complex64），由振幅和相位构造
        可根据需要扩展其他字段。
        """
        sample = self.samples[idx]

        # 构建 LDI 输入：对于 0..active_max_ldi_layer，交替拼接 img 和 depth
        rgbd_parts = []
        for i in range(self.active_max_ldi_layer + 1):
            img_key = f"img_{i}"
            depth_key = f"depth_{i}"
            img = sample[img_key]  # shape (3, H, W)
            depth = sample[depth_key]  # shape (1, H, W)
            rgbd_parts.append(img)
            rgbd_parts.append(depth)

        # 沿通道维拼接，最终形状 (4*(active_max_ldi_layer+1), H, W)
        rgbd = np.concatenate(rgbd_parts, axis=0)

        # 目标全息图（原始标签即为 amp_4, phs_4）
        amp_4 = sample["amp_4"]   # (3, H, W)
        phs_4 = sample["phs_4"]   # (3, H, W)

        # 转换为 torch 张量
        amp = torch.from_numpy(amp_4).float()
        phs = torch.from_numpy(phs_4).float()

        # 将归一化相位映射到 [-π, π]
        phs_scaled = (phs - 0.5) * 2 * math.pi

        # 构造复数目标
        target_complex = torch.polar(amp, phs_scaled)  # 直接得到复数张量

        return {
            "rgbd": torch.from_numpy(rgbd).float(),
            "amp_4": amp,
            "phs_4": phs,
            "target_complex": target_complex,
        }


def create_dataloader(
    tfrecord_path: str,
    dataset_params: Dict[str, Any],
    labels: List[str],
    active_max_ldi_layer: int = 0,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """
    工厂函数，根据原项目参数快速构建 PyTorch DataLoader。

    Args:
        tfrecord_path: TFRecord 文件路径。
        dataset_params: 数据集参数（res_h, res_w, sample_count 等）。
        labels: 特征标签列表。
        active_max_ldi_layer: LDI 层数。
        batch_size: 批量大小。
        shuffle: 是否打乱数据。
        num_workers: 加载进程数。
        drop_last: 是否丢弃最后不足一批的数据。

    Returns:
        torch.utils.data.DataLoader 实例。
    """
    dataset = THDataset(
        tfrecord_path=tfrecord_path,
        dataset_params=dataset_params,
        labels=labels,
        active_max_ldi_layer=active_max_ldi_layer,
        load_to_memory=True
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
    )