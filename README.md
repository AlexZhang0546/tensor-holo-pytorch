# Tensor Holography PyTorch

本项目是基于 PyTorch 实现的 **张量全息术（Tensor Holography）** 完整训练与推理流水线，源自原 TensorFlow 实现的迁移与重构。支持端到端的全息图预测、两阶段训练（主网络 + DDPM 校正）、批量验证、单张图像评估以及 ONNX 导出，便于 TensorRT 部署。

## 主要特性

- **两阶段训练**：
  - **Stage 1**：训练主网络 `TensorHolographyNet`，使用全息图损失 + 焦栈感知损失 + TV 损失。
  - **Stage 2**：加载 Stage 1 权重，引入 DDPM 校正网络，先进行恒等预训练，再联合微调；支持深度偏移与填充。
- **完整光学仿真**：角谱传播、三种双相位编码（AA-DPM / BL-DPM / Maimone DPM）、物理光圈滤波。
- **评估与部署**：批量验证（SSIM/PSNR）、单张 RGB+深度 图像推理、导出 ONNX 模型。
- **纯 PyTorch 实现**：所有组件（包括复数运算、传播、变换）均以 PyTorch 编写，可微分且兼容 ONNX/TensorRT。

## 环境与安装

### 依赖库

- Python 3.8+
- PyTorch >= 1.12
- NumPy
- OpenCV-Python
- TensorBoard (可选)
- `tfrecord` 库 (用于读取 TFRecord 数据)

### 安装步骤

```bash
# 克隆仓库
git clone <repository-url>
cd tensor-holography-pytorch

# 安装核心依赖（推荐使用虚拟环境）
pip install torch numpy opencv-python tensorboard tfrecord
```

## 数据准备

项目使用原 TensorFlow 项目中的 **TFRecord** 格式数据。数据文件应放置在 `data/` 目录下，结构如下：

```
data/
├── train_192_v2/
│   └── train_04.tfrecord       # 训练集（约 3800 样本）
├── test_192_v2/
│   └── test_04.tfrecord        # 验证集（约 100 样本）
└── validate_192_v2/
    └── validate_04.tfrecord    # 验证集（可与其他共用）
```

TFRecord 文件中包含以下特征（与 `labels` 参数对应）：
- `amp_4`：目标振幅（3 通道）
- `phs_4`：目标相位（3 通道）
- `img_0`：RGB 图像（3 通道）
- `depth_0`：深度图（1 通道）
（多层 LDI 时会包含 `img_1`, `depth_1` 等）

> 如果数据路径或文件名不同，请在训练/验证命令中调整相应参数（暂不支持配置文件，直接修改代码或使用软链接）。

## 使用方法

所有命令均通过主入口 `main.py` 调用，支持子命令：`train_stage1`, `train_stage2`, `validate`, `evaluate`, `export`。

### 1. Stage 1 训练（主网络）

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
    --ckpt-dir ./ckpt_stage1
```

- `--restore`：从已有 checkpoint 恢复训练（需存在 `ckpt_dir/stage1_latest.pth`）。
- 默认数据路径为 `data/train_192_v2/train_04.tfrecord` 和 `data/test_192_v2/test_04.tfrecord`。如需更改，可直接修改 `src/train/stage1.py` 中的路径变量。

### 2. Stage 2 训练（DDPM 校正）

Stage 2 需要先完成 Stage 1 并获得模型权重文件。

```bash
python main.py train_stage2 \
    --stage1-ckpt ./ckpt_stage1/stage1_latest.pth \
    --activate-ddpm \
    --depth-shift 12.0 \
    --padding 0 \
    --joint-epochs 200 \
    --stage2-ckpt-dir ./ckpt_stage2 \
    --restore-stage1
```

- 若不使用 DDPM，可添加 `--bypass-ddpm-network`（此时仅训练主网络微调）。
- `--restore-stage2`：从 `stage2_ckpt_dir` 中恢复之前的 Stage 2 训练（优先恢复联合阶段，否则恢复恒等预训练）。
- Stage 2 会先进行恒等预训练（默认 50 epoch），再联合训练（默认 200 epoch）。可通过 `--stage2-epochs` 和 `--joint-epochs` 调整。

### 3. 验证（Stage 1 或 Stage 2）

```bash
# 验证 Stage 1 模型
python main.py validate --mode stage1 --ckpt-path ./ckpt_stage1/stage1_latest.pth

# 验证 Stage 2 模型（含 DDPM）
python main.py validate --mode stage2 \
    --ckpt-path ./ckpt_stage2/stage2_joint_latest.pth \
    --activate-ddpm \
    --padding 0 \
    --depth-shift 12.0
```

验证脚本会在验证集上计算振幅图的 SSIM 和 PSNR（Stage 1 直接比较输出；Stage 2 经过完整传播、DDPM、编码和滤波后与目标比较）。

### 4. 单张图像评估（推理）

对任意 RGB 和深度图生成全息图相位：

```bash
python main.py evaluate \
    --ckpt-path ./ckpt_stage2/stage2_joint_latest.pth \
    --activate-ddpm \
    --eval-rgb-path /path/to/rgb.png \
    --eval-depth-path /path/to/depth.png \
    --eval-output-path ./output \
    --eval-res-h 1080 \
    --eval-res-w 1920 \
    --padding 0 \
    --use-maimone-dpm \
    --adaptive-phs-shift
```

- 支持多种双相位编码：`--use-maimone-dpm`, `--use-bldpm`（默认 AA-DPM）。
- 可调节 `--phs-max`, `--gaussian-sigma` 等参数。
- 输出文件：`amp.png`, `phs.png`, `blue.png` (B通道相位), `green.png`, `red.png`, `amp_filtered.png`。

### 5. 导出 ONNX

将训练好的模型（含 DDPM 可选）导出为 ONNX 格式，用于 TensorRT 部署。

```bash
python main.py export \
    --ckpt-path ./ckpt_stage2/stage2_joint_latest.pth \
    --activate-ddpm \
    --output ./model.onnx \
    --res-h 1080 \
    --res-w 1920 \
    --pad 0
```

- 导出时固定 `depth_shift=0`（ONNX 不支持复数传播），仅输出网络预测的振幅和相位。
- 如需包含 DDPM，务必添加 `--activate-ddpm` 并确保 checkpoint 中含有 DDPM 权重。

## 项目结构

```
tensor-holography-pytorch/
├── main.py                      # 程序入口，解析子命令并分发
├── configs/                     # 配置文件（预留，暂未使用）
├── data/                        # 数据目录（需用户自行放置 TFRecord）
│   ├── train_192_v2/
│   ├── test_192_v2/
│   └── validate_192_v2/
├── src/
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset，读取 TFRecord
│   │   └── transforms.py        # interleave/deinterleave
│   ├── eval/
│   │   ├── evaluate.py          # 单张图像推理
│   │   ├── export_onnx.py       # 导出 ONNX
│   │   └── validate.py          # 批量验证
│   ├── losses/
│   │   ├── ddpm_loss.py         # DDPM 相位统计正则
│   │   ├── focal_stack.py       # 焦栈感知损失
│   │   └── holo_loss.py         # 振幅-相位损失
│   ├── models/
│   │   ├── holonet.py           # 主网络 TensorHolographyNet
│   │   └── ddpm_net.py          # DDPM 校正网络
│   ├── optics/
│   │   ├── aperture.py          # 物理光圈滤波
│   │   ├── complex_utils.py     # 复数运算、FFT 辅助
│   │   ├── dpm.py               # 三种双相位编码
│   │   └── propagation.py       # 角谱传播算子
│   ├── train/
│   │   ├── stage1.py            # Stage 1 训练脚本
│   │   ├── stage2.py            # Stage 2 训练脚本
│   │   └── trainer.py           # 训练器基类（预留）
│   └── utils/
│       ├── metrics.py           # SSIM / PSNR
│       ├── visualizer.py        # 可视化工具（暂空）
│       └── weight_init.py       # 权重初始化
└── README.md
```

## 注意事项

- 所有光学仿真（传播、DPM、滤波）均基于 **NCHW** 张量格式。
- 默认数据分辨率为 192×192（训练），推理时支持任意分辨率（需确保模型结构可适应）。
- Stage 2 的深度偏移 `depth_shift` 需与训练时保持一致，否则结果不匹配。
- 导出 ONNX 时仅支持 `depth_shift=0`，因为复数传播无法在 ONNX 中直接表示。

## 引用

如果本代码对您的研究有帮助，请引用原始 TensorFlow 项目以及本 PyTorch 迁移版本（如有相关论文，请补充）。

---

**License**：本项目基于 [原始 TensorFlow 代码](https://github.com/... ) 迁移，遵循其许可证。请自行确认。

## 联系

如有问题或建议，欢迎提交 Issue 或 Pull Request。