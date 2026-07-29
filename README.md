# Tensor Holography PyTorch

本项目是 **Tensor Holography** 的 PyTorch 移植实现，能够从 RGB‑D 图像（或 LDI 多层深度图像）生成高质量的计算全息图。  
训练分为两个阶段：主网络（HoloNet）预训练和 DDPM 精细调整，支持多种双相位编码方法（AA‑DPM / BL‑DPM / Maimone DPM）以及物理孔径滤波，最终可导出 ONNX 模型用于部署加速。

---

## 主要特性

- **纯 PyTorch 实现**：无 TensorFlow 依赖，仅使用 `tfrecord` 包读取原始数据。
- **两阶段训练**  
  - Stage 1：训练 HoloNet，组合全息损失 + 焦栈损失 + TV 损失。  
  - Stage 2：冻结 / 联合优化 DDPM 网络，进一步提升图像质量。
- **完整的光学仿真**：角谱传播、深度偏移、双相位编码、频域孔径滤波。
- **多种双相位编码**：Anti‑Aliasing DPM、Band‑Limited DPM、原始 Maimone DPM。
- **便捷的评估与导出**  
  - 验证集批量评估 SSIM / PSNR。  
  - 单张 RGB‑D 推理，输出相位图、振幅图。  
  - 导出 ONNX 模型，供 TensorRT 等推理框架使用。

---

## 项目结构

```
.
├── main.py                  # 总入口，命令行参数分发
├── src/
│   ├── data/
│   │   ├── dataset.py       # TFRecord 数据集读取
│   │   └── transforms.py    # interleave/deinterleave
│   ├── models/
│   │   ├── holonet.py       # 主全息预测网络
│   │   └── ddpm_net.py      # DDPM 校正网络
│   ├── optics/
│   │   ├── propagation.py   # 角谱 / 菲涅尔传播
│   │   ├── complex_utils.py # 复数构造、FFT 工具
│   │   ├── aperture.py      # 孔径滤波
│   │   └── dpm.py           # 双相位编码
│   ├── losses/
│   │   ├── holo_loss.py     # 全息振幅‑相位损失
│   │   ├── focal_stack.py   # 焦栈感知损失
│   │   └── ddpm_loss.py     # DDPM 相位正则
│   ├── train/
│   │   ├── stage1.py        # 阶段一训练
│   │   ├── stage2.py        # 阶段二训练
│   │   └── trainer.py       # 训练基类
│   ├── eval/
│   │   ├── validate.py      # 批量验证
│   │   ├── evaluate.py      # 单张推理评估
│   │   └── export_onnx.py   # 导出 ONNX
│   └── utils/
│       ├── metrics.py       # SSIM / PSNR
│       ├── weight_init.py   # 权重初始化
│       └── visualizer.py    # （预留）
├── data/                    # 数据集目录（需自行准备）
└── model/                   # 模型保存目录
```

---

## 环境配置

- Python 3.8+
- PyTorch ≥ 1.10（推荐 2.0+）
- CUDA（可选，但建议使用 GPU）

### 安装依赖

```bash
pip install torch torchvision  # 根据 CUDA 版本选择
pip install numpy opencv-python tfrecord
```

`tfrecord` 包用于直接读取 TensorFlow 的 TFRecord 文件，无需安装 TensorFlow。

---

## 数据准备

训练数据需为 TFRecord 格式，每条样本包含以下特征（`float_list`）：

| 特征键      | 形状          | 说明                     |
|-------------|---------------|--------------------------|
| `amp_4`     | (3, H, W)     | 目标振幅，范围 [0, √2]  |
| `phs_4`     | (3, H, W)     | 目标相位，归一化到 [0,1] |
| `img_0`     | (3, H, W)     | 输入 RGB 图像，[0,255] 或归一化后（代码内会除以 255） |
| `depth_0`   | (1, H, W)     | 输入深度图，值域 [0,1]   |
| `img_1`, `depth_1`, … | 同上 | 多层 LDI 时使用       |

推荐目录结构：

```
data/
  train_192_v2/
    train_04.tfrecord
  test_192_v2/
    test_04.tfrecord
  validate_192_v2/
    validate_04.tfrecord
```

在训练脚本中，可通过 `--dataset-res`、`--active-max-ldi-layer` 等参数指定分辨率和 LDI 层数，程序会自动拼接路径。

---

## 使用方法

所有功能通过 `main.py` 提供统一入口：

```bash
python main.py <mode> [options...]
```

支持的 mode：  
`train_stage1` | `train_stage2` | `validate` | `evaluate` | `export`

### 1. 训练阶段一（HoloNet）

```bash
python main.py train_stage1 \
  --model-name full_loss \
  --dataset-res 192 \
  --pitch 0.008 \
  --num-layers 30 \
  --num-filters-per-layer 24 \
  --num-epochs 4050 \
  --batch 2 \
  --learning-rate 1e-4 \
  --num-iter-per-test 1000 \
  --active-max-ldi-layer 0 \
  [--restore] [--ckpt-dir ./model/my_stage1]
```

**关键参数：**

- `--dataset-res`：数据集分辨率（高宽相同，如 192）。
- `--pitch`：像素间距（mm）。
- `--num-layers` / `--num-filters-per-layer`：网络深度与宽度。
- `--batch`：批次大小。
- `--active-max-ldi-layer`：LDI 层数，0 表示单层 RGB‑D，>0 表示多层。
- `--restore`：若指定，从 `ckpt-dir` 下的 `stage1_latest.pth` 恢复训练。
- `--ckpt-dir`：模型保存目录，默认自动生成。

### 2. 训练阶段二（DDPM 精细调整）

```bash
python main.py train_stage2 \
  --model-name full_loss \
  --dataset-res 192 \
  --pitch 0.008 \
  --num-layers 30 \
  --num-filters-per-layer 24 \
  --batch 2 \
  --learning-rate 1e-4 \
  --stage1-ckpt ./model/ckpt_full_loss_.../stage1_latest.pth \
  --stage2-epochs 50 \
  --joint-epochs 200 \
  --padding 0 \
  --depth-shift 12.0 \
  [--activate-ddpm] [--bypass-ddpm-network] \
  [--restore-stage1] [--restore-stage2] [--stage2-ckpt-dir ./model/my_stage2]
```

**关键参数：**

- `--stage1-ckpt`：阶段一的 checkpoint 路径（必须）。
- `--padding`：边缘填充大小，用于模拟衍射超出区域。
- `--depth-shift`：深度偏移距离（mm）。
- `--stage2-epochs`：身份预训练轮数（仅优化 DDPM）。
- `--joint-epochs`：联合训练轮数（同时优化 HoloNet 和 DDPM）。
- `--activate-ddpm`：启用 DDPM 网络；不启用则等效于 `--bypass-ddpm-network`。
- `--restore-stage2`：从 `stage2_identity_latest.pth` 或 `stage2_joint_latest.pth` 恢复训练。

### 3. 验证

```bash
# 验证 stage1 模型
python main.py validate --mode stage1 --ckpt-path ./model/.../stage1_latest.pth

# 验证 stage2 模型（可选 DDPM）
python main.py validate --mode stage2 \
  --ckpt-path ./model/.../stage2_joint_latest.pth \
  --padding 0 --depth-shift 12.0 \
  [--activate-ddpm] [--bypass-ddpm-network] [--ddpm-ckpt-path ...]
```

程序会在验证 TFRecord 上计算振幅图的 SSIM / PSNR 并输出统计信息。

### 4. 单张图像评估

```bash
python main.py evaluate \
  --ckpt-path ./model/stage2_joint_latest.pth \
  --eval-rgb-path ./test_img/input.png \
  --eval-depth-path ./test_img/depth.png \
  --eval-output-path ./results/ \
  --eval-res-h 1080 --eval-res-w 1920 \
  --padding 0 --eval-depth-shift 0.0 \
  --use-aadpm  # 或 --use-maimone-dpm / --use-bldpm
```

**关键参数：**

- `--eval-rgb-path` / `--eval-depth-path`：输入 RGB 和深度图路径，深度图自动以灰度读取。
- `--eval-output-path`：输出目录，会生成 `amp.png`、`phs.png`、各通道相位图等。
- `--eval-depth-shift`：推理时的额外深度偏移。
- `--use-maimone-dpm` / `--use-bldpm`：选择双相位编码方法；默认使用 AA‑DPM。
- `--phs-max`：相位包裹上限（默认 `2.0`，即 2π）。
- `--gaussian-sigma`：AA‑DPM 的模糊程度。

### 5. 导出 ONNX

```bash
python main.py export \
  --ckpt-path ./model/stage2_joint_latest.pth \
  --output model.onnx \
  --res-h 1080 --res-w 1920 --pad 0 \
  [--activate-ddpm] [--ddpm-ckpt-path ...]
```

导出的 ONNX 包含两个输出 `amp_out` 和 `phs_out`，batch 维度动态。  
**注意**：导出时不包含复数深度偏移运算，需在后续的推理管线中补充。

---

## 主要参数速查表

| 参数                         | 类型    | 默认值      | 说明                                 |
|------------------------------|---------|-------------|--------------------------------------|
| `--model-name`               | str     | `full_loss` | 模型名称，用于自动生成保存目录         |
| `--dataset-res`              | int     | 192         | 数据集分辨率（高宽相等）              |
| `--pitch`                    | float   | 0.008       | 像素尺寸（mm）                        |
| `--num-layers`               | int     | 30          | HoloNet 层数                          |
| `--num-filters-per-layer`    | int     | 24          | 每层滤波器数                          |
| `--batch`                    | int     | 2           | 批次大小                              |
| `--learning-rate`            | float   | 1e-4        | 学习率                                |
| `--active-max-ldi-layer`     | int     | 0           | 最大 LDI 层索引（0 为单层 RGB‑D）      |
| `--padding`                  | int     | 0           | 边缘填充                              |
| `--depth-shift`              | float   | 12.0        | 深度偏移（mm）                        |
| `--activate-ddpm`            | flag    | False       | 启用 DDPM 网络                        |
| `--bypass-ddpm-network`      | flag    | False       | 旁路 DDPM 网络（即使存在也不使用）    |
| `--num-iter-per-test`        | int     | 1000        | 训练期间验证间隔（step 数）           |
| `--restore` / `--restore-stage1` / `--restore-stage2` | flag | False | 从检查点恢复训练                |
| `--ckpt-path` / `--stage1-ckpt` 等 | str | required | 模型权重路径                         |

---

## 常见问题

**Q：数据读取时提示 `No module named 'tfrecord'`？**  
A：请执行 `pip install tfrecord`。该包不依赖 TensorFlow。

**Q：ONNX 模型能在 TensorRT 中使用吗？**  
A：可以。导出时已选用 opset 14，建议使用 `trtexec` 或 ONNX Runtime 测试。注意深度偏移等光学后处理仍需在外部实现。

**Q：训练时 GPU 内存不足？**  
A：减小 `--batch`，或降低 `--dataset-res`（如 128）。也可尝试减小 `--num-filters-per-layer`。

---

## 引用

本实现参考了以下工作：

- Shi, L., Li, B., Kim, C., Kellnhofer, P., & Matusik, W. (2021).  
  **Towards real-time photorealistic 3D holography with deep neural networks**. *Nature*.  
- Sui, X., He, Z., Chu, D., & Cao, L. (2021). **Band-limited double-phase method for holographic displays**.  
- Maimone, A., Georgiou, A., & Kollin, J. S. (2017). **Holographic near-eye displays for virtual and augmented reality**.

---

## 许可证

本项目代码基于原 TensorFlow 实现移植，仅供研究使用。  
数据及模型权重请遵循原始仓库的许可协议。