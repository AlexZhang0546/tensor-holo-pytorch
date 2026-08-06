# Tensor Holography（PyTorch 移植版）

## 简介

本仓库是 **Tensor Holography V2**（*End-to-end Learning of 3D Phase-only Holograms for
Holographic Display*，Shi et al., Light: Science and Applications, 2022）的 PyTorch 移植实现。
原版为 TensorFlow 1.x 实现（`main_v2.py`，幅值 + 相位双路输出），本移植的主要差异：

- 网络直接输出**复数光场** `(B, 3, H, W)`（complex64），替代原版的幅值/相位两路输出；
- 全链路（网络、光学传播、双相位编码、孔径滤波、损失）均为纯 PyTorch 可微实现，无 TensorFlow 依赖，
  仅使用 `tfrecord` 包直接读取原始 TFRecord 数据；
- CLI 参数名与默认值尽量与 `main_v2.py` 保持一致（单 argparse + 模式开关）。

主要功能：

- **两阶段训练**：Stage 1 训练主网络 `ComplexHoloNet`（复数全息损失 + 焦栈损失 + TV 损失）；
  Stage 2 先做 DDPM 恒等预训练，再做主网络 + DDPM 联合微调（焦栈 + TV + 相位统计正则）。
- **光学仿真**：角谱传播（支持 `double_pad`）、深度偏移、AA-DPM / BL-DPM / Maimone DPM
  三种双相位编码、物理孔径滤波。
- **评估与部署**：验证集批量 SSIM/PSNR；单张 RGB-D 推理输出相位图/振幅图；导出 ONNX
  （输出拆为实部/虚部两个实数张量）。

## 项目结构

```text
.
├── main.py                  # 总入口：原始风格参数分发
├── src/
│   ├── data/
│   │   ├── dataset.py       # TFRecord 数据集读取（与原 _extract_fn 解析一致）
│   │   └── transforms.py    # interleave / deinterleave
│   ├── models/
│   │   ├── holonet.py       # ComplexHoloNet 主全息预测网络（复数）
│   │   ├── ddpm_net.py      # ComplexDDPMNet 校正网络（复数）
│   │   └── complex_layers.py# 复数卷积 / BN / ReLU 等基础模块
│   ├── optics/
│   │   ├── propagation.py   # 角谱（AS）/ 菲涅尔传播
│   │   ├── complex_utils.py # 复数构造、FFT 工具
│   │   ├── dpm.py           # AA-DPM / BL-DPM / Maimone DPM 双相位编码
│   │   └── aperture.py      # 物理孔径滤波
│   ├── losses/
│   │   ├── complex_losses.py# 复数全息损失（复数域）
│   │   ├── focal_stack.py   # 焦栈感知损失（含 SSIM/PSNR 指标）
│   │   ├── holo_loss.py     # 全息图振幅-相位损失（保留）
│   │   └── ddpm_loss.py     # DDPM 相位均值/标准差正则
│   ├── train/
│   │   ├── stage1.py        # 阶段一训练（主网络）
│   │   ├── stage2.py        # 阶段二训练（identity + joint）
│   │   └── trainer.py       # 训练基类（保留）
│   ├── eval/
│   │   ├── validate.py      # 批量验证（stage1 / stage2）
│   │   ├── evaluate.py      # 单张 RGB-D 推理
│   │   └── export_onnx.py   # 导出 ONNX
│   └── utils/
│       ├── metrics.py       # SSIM / PSNR（替代 tf.image.ssim / psnr）
│       ├── weight_init.py   # 权重初始化
│       └── visualizer.py    # 可视化（预留）
├── data/                    # 数据集目录（TFRecord，需自行准备）
├── model/                   # checkpoint 默认保存目录
└── requirements.txt
```

## 环境配置

依赖（`requirements.txt`）：

```text
torch>=1.12.0
numpy>=1.21.0
opencv-python>=4.5.0
tfrecord>=1.0.0
protobuf>=3.20.0
tensorboard>=2.9.0   # 可选，用于训练日志记录
```

`tfrecord` 包用于直接读取原始 TFRecord 文件，无需安装 TensorFlow。

本项目实际使用的服务器运行环境：

- 项目路径：`/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch`
- 虚拟环境：`conda activate holography`
- 长时训练建议在 `screen` 会话中运行（例如 `screen -S zrx-tensor-holo` 新建、
  `screen -r zrx-tensor-holo` 恢复）。

## 数据准备

数据为 TFRecord 格式（TensorHolo V2 数据集，从原仓库 Data 链接下载后将 `*_384_v2` 等
子目录放入 `data/`）。每条样本为 `float_list` 特征：

| 特征键 | 形状 | 说明 |
|--------|------|------|
| `amp_4` | (3, H, W) | 目标振幅，值域 [0, √2] |
| `phs_4` | (3, H, W) | 目标相位，归一化到 [0, 1]（构造复数目标时映射到 [-π, π]） |
| `img_i` | (3, H, W) | 第 i 层 LDI 的 RGB 图像 |
| `depth_i` | (1, H, W) | 第 i 层 LDI 的深度图 |

代码按以下约定自动拼接 TFRecord 路径（`res` = `--dataset-res`，`L` = `--active-max-ldi-layer`）：

- Stage 1 训练：`data/train_{res}_v2/train_{L}4.tfrecord`；训练中验证：`data/test_{res}_v2/test_{L}4.tfrecord`
- Stage 2 训练（固定单层 RGBD 输入）：`data/train_{res}_v2/train_04.tfrecord`；
  训练中验证：`data/test_{res}_v2/test_04.tfrecord`
- `validate` 模式：`data/validate_{res}_v2/validate_04.tfrecord`

服务器上已有的数据目录：`data/train_384_v2`、`data/test_384_v2`、`data/validate_384_v2`。

## 使用方法

所有功能通过 `main.py` 提供统一入口（参照 `main_v2.py` 的单 argparse 风格）：

- `--train-mode --train-stage stage1|stage2`：训练 Stage 1 / Stage 2。
- `--validate-mode-s1` / `--validate-mode-s2`：验证。
- `--eval-mode`：单张 RGB-D 推理。
- `--export-mode`：导出 ONNX。
- `--dry-run`：只打印翻译后的命令，不执行。

### 训练 Stage 1（主网络）

```bash
python main.py --train-mode --train-stage stage1 \
  --dataset-res 384 --num-epochs 200 --batch 2 --learning-rate 1e-4 \
  --num-iter-per-test 200
```

- 默认 checkpoint 目录：
  `model/ckpt_{model-name}_pitch_{pitch*1000}_layers_{num-layers}_filters_{num-filters}_stage1/`
- 保存 `stage1_epoch_{epoch:04d}.pth` 与 `stage1_latest.pth`。
- 断点续训：加 `--restore`（`--ckpt-dir` 指定目录时从该目录的 `stage1_latest.pth` 恢复）。

### 训练 Stage 2（identity 预训练 + joint 联合训练）

```bash
python main.py --train-mode --train-stage stage2 \
  --dataset-res 384 --activate-ddpm \
  --restore-stage1 \
  --stage1-ckpt model/ckpt_full_loss_pitch_8_layers_30_filters_24_stage1/stage1_latest.pth \
  --stage2-ckpt-dir model/stage2_v2 \
  --stage2-epochs 50 --joint-epochs 200 \
  --depth-shift 12.0 --num-iter-per-test 200
```

流程：先运行 `--stage2-epochs` 个 **identity 预训练** epoch（主网络冻结，仅训练 DDPM，
使输出场≈输入场），再运行 `--joint-epochs` 个 **联合训练** epoch（主网络 + DDPM 同时训练）。

- `--train-depth-shift` 是 `--depth-shift` 的别名（深度偏移，mm）。
- `--activate-ddpm`：启用 DDPM 网络；`--bypass-ddpm-network`：旁路 DDPM（仅训练主网络，
  通常用于 0 mm 偏移）。
- 默认 checkpoint 目录：
  `model/ckpt_{model-name}_pitch_{pitch*1000}_layers_{num-layers}_filters_{num-filters}_ddpm_{shift}/`
  （`--bypass-ddpm-network` 时追加 `_bypass` 后缀）。

Stage 2 续训（优先从 joint checkpoint 恢复、自动跳过 identity）：

```bash
python main.py --train-mode --train-stage stage2 \
  --dataset-res 384 --activate-ddpm \
  --restore-stage2 --stage2-ckpt-dir model/stage2_v2 \
  --joint-epochs 400 --depth-shift 12.0
```

`--restore-stage2` 优先加载 `stage2_joint_latest.pth`（含主网络 + DDPM + 优化器），从
`epoch + 1` 继续；若只有 identity checkpoint 则从 identity 继续。

### 验证

```bash
python main.py --validate-mode-s1 --ckpt-path model/.../stage1_latest.pth
python main.py --validate-mode-s2 --activate-ddpm \
  --ckpt-path model/.../stage2_joint_latest.pth --depth-shift 12.0
```

- Stage 1 验证：主网络输出复数场，比较预测振幅与目标振幅的 SSIM / PSNR。
- Stage 2 验证：走完整物理链路（深度偏移 → DDPM → AA-DPM → 孔径滤波 → 反向传播），
  比较重建场振幅与目标振幅的 SSIM / PSNR。

### 单张图像评估

```bash
python main.py --eval-mode --activate-ddpm \
  --ckpt-path model/.../stage2_joint_latest.pth \
  --eval-rgb-path data/example_input/couch_rgb.png \
  --eval-depth-path data/example_input/couch_depth.png \
  --eval-output-path output/ \
  --eval-res-h 1080 --eval-res-w 1920 \
  --depth-shift 12.0 --eval-depth-shift 0.0
```

- 输出：相位图、振幅图及各通道相位图等。
- 双相位编码默认使用 AA-DPM；可切换 `--use-maimone-dpm` / `--use-bldpm`。
- `--gaussian-sigma` / `--gaussian-width`：AA-DPM 预模糊参数。
- `--phs-max`：SLM 最大相位调制（单位 π，默认 `2.0`，即 2π）。

### 导出 ONNX

```bash
python main.py --export-mode --activate-ddpm \
  --ckpt-path model/.../stage2_joint_latest.pth \
  --trt-res-h 1080 --trt-res-w 1920 --output model.onnx
```

ONNX 不支持复数张量与 FFT/IFFT，因此导出时把网络输出拆为 `real_out` / `imag_out`
两个实数张量（opset 18，batch 维度动态）。深度偏移、双相位编码等光学后处理需在
推理管线中自行实现。

### ONNX 正确性测试

```bash
# stage1（无 DDPM）
python test_onnx.py --ckpt model/.../stage1_latest.pth --res 384

# stage2（holonet + DDPM）
python test_onnx.py --ckpt model/.../stage2_joint_latest.pth --activate-ddpm --res 384
```

脚本会导出 ONNX，并在相同输入上对比 PyTorch 与 ONNX Runtime 的 `real_out` / `imag_out`
（最大/平均绝对误差、相对误差、振幅 SSIM），batch 1/2 均通过且误差在 1e-5 量级即视为 PASS。
数值对比测试通过后，导出时的分辨率（`--trt-res-h/w`）决定部署分辨率，两者相互独立。

## 主要参数速查表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset-res` | 192 | 数据集分辨率（高宽相同） |
| `--pitch` | 0.008 | 像素尺寸（mm） |
| `--num-layers` | 30 | HoloNet 层数 |
| `--num-filters-per-layer` | 24 | 每层滤波器数 |
| `--model-name` | `full_loss` | 模型名（用于 checkpoint 目录命名） |
| `--num-epochs` | 4050 | Stage 1 训练总 epoch 数 |
| `--epoch_to_start_ddpm_training` | 3000 | 信息性参数（移植版中训练已拆分为两个脚本） |
| `--stage2-epochs` | 50 | Stage 2 identity 预训练 epoch 数 |
| `--joint-epochs` | 200 | Stage 2 联合训练 epoch 数 |
| `--batch` | 2 | 批大小 |
| `--learning-rate` | 1e-4 | 学习率 |
| `--num-iter-per-test` | 1000 | 训练中验证间隔（step） |
| `--active-max-ldi-layer` | 0 | 最大 LDI 层索引（0 = 单层 RGBD） |
| `--depth-shift` / `--train-depth-shift` | 12.0 | Stage 2 深度偏移（mm） |
| `--padding` | 0 | 全息图边缘填充（容纳出画幅衍射） |
| `--activate-ddpm` | False | 启用 DDPM 网络 |
| `--bypass-ddpm-network` | False | 旁路 DDPM 网络 |
| `--eval-res-h` / `--eval-res-w` | 1080 / 1920 | 评估模式输入分辨率 |
| `--eval-depth-shift` | 0.0 | 推理时相对中点全息面的深度偏移（mm） |
| `--trt-res-h` / `--trt-res-w` | 1080 / 1920 | 导出模式输入分辨率 |
| `--phs-max` | 2.0 | SLM 最大相位调制（单位 π） |
| `--gaussian-sigma` / `--gaussian-width` | 0.0 / 3 | AA-DPM 预模糊参数 |
| `--k` | 1.0 | BL-DPM 频域掩码参数 |

## 指标说明

- 训练日志中 identity 阶段打印的 `SSIM`：DDPM 输出场与输入（深度偏移后）复数场的**振幅相似度**
  （恒等对齐度），不是重建图像质量。
- Stage 2 joint 阶段每 step 打印的 `SSIM_amp`：DPM 重建场振幅与目标振幅的 SSIM；
  验证时同时打印 `SSIM_amp` 与 `SSIM_img`。
- `SSIM_img`：焦栈重建图像的 SSIM（`compute_focal_stack_loss` 内的 `ssim_img_loss`），
  即论文报告的重建图像质量指标。
- `compute_ssim` 使用 11×11 高斯窗、σ=1.5；`data_range` 默认 √2（焦栈图像用 1.0，
  Stage 2 振幅验证用 1.414）。

## 实现说明（与原 TensorFlow 版的差异）

- **复数场直出**：原版为幅值 + 相位两路输出，本移植为复数场 `(B, 3, H, W)` 直接输出。
- **`depth_to_space` 对齐**：TF 的 `depth_to_space(NCHW)` 通道→空间映射与 PyTorch
  `pixel_shuffle` 不同，`src/optics/dpm.py` 实现了 `_depth_to_space_nchw` 保持与原版一致。
- **DDPM 输出振幅限幅**：`src/models/ddpm_net.py` 将输出振幅限制在 √2（对应原版 tanh
  输出上界），避免离群尖峰把 DPM 的 `amp_max` 顶得过大。
- **复数 BatchNorm**：`ComplexBatchNorm2d` 的协方差统计 `V_ri` 初始为 0
  （`src/models/complex_layers.py`），保证 eval 模式下数值稳定。
- **波长**：默认 `[450, 520, 638] nm`；深度映射 `depth_base=-3`、`depth_scale=6`。

## 引用与许可

本移植参照以下工作（引用信息来自原仓库 README）：

- Shi, L., Li, B., Kim, C., Kellnhofer, P., & Matusik, W. (2021).
  *Towards real-time photorealistic 3D holography with deep neural networks*. Nature.
- Shi, L., Li, B., & Matusik, W. (2022).
  *End-to-end learning of 3D phase-only holograms for holographic display*.
  Light: Science and Applications.
- Sui, X., He, Z., Chu, D., & Cao, L. (2021). *Band-limited double-phase method
  for holographic displays*.
- Maimone, A., Georgiou, A., & Kollin, J. S. (2017). *Holographic near-eye displays
  for virtual and augmented reality*.

数据与模型权重请遵循原始仓库（MIT 技术许可办公室自定义许可）的许可协议。
