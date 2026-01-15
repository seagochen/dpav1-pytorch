# Depth Anything PyTorch

基于 DINOv2 的单目深度估计模型 PyTorch 实现。

> 本项目仅供学习研究使用，代码参考了 [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) 官方实现。

## 项目简介

Depth Anything 是一个强大的单目深度估计模型，采用 DINOv2 Vision Transformer 作为编码器，DPT (Dense Prediction Transformer) 作为解码器，能够从单张 RGB 图像预测相对深度图。

本项目对原始代码进行了模块化重构，便于学习和理解模型架构。

## 示例

| 输入图像 | 深度估计结果 |
|:---:|:---:|
| ![Input](assets/hamburg.png) | ![Depth](assets/hamburg_depth.png) |

## 模型架构

```
输入图像 (H×W×3)
    ↓
DINOv2 编码器 (ViT-Small/Base)
    ↓
多尺度特征 (4个层级)
    ↓
DPT 解码器 (特征融合 + 上采样)
    ↓
深度图 (H×W×1)
```

## 支持的模型

| 模型 | 参数量 | 说明 |
|------|--------|------|
| vits | 24.8M | ViT-Small，快速推理 |
| vitb | 97.5M | ViT-Base，均衡性能 |

## 安装

```bash
git clone https://github.com/your-username/dpav1-pytorch.git
cd dpav1-pytorch
pip install -r requirements.txt
```

## 使用方法

### Python API

```python
from depth_anything import DepthAnything, DepthTransform
import cv2

# 加载模型
model = DepthAnything.from_pretrained('vitb')
model.eval().cuda()

# 预处理
transform = DepthTransform(size=518)
image = cv2.imread('image.jpg')
tensor = transform(image).cuda()

# 推理
depth = model.predict(tensor)
```

### 命令行

```bash
# 单张图像
python scripts/infer.py --encoder vitb --input image.jpg --output output/

# 批量处理
python scripts/infer.py --encoder vits --input images/ --output output/

# 仅保存深度图 (不保存对比图)
python scripts/infer.py --encoder vitb --input image.jpg --pred-only

# 灰度输出
python scripts/infer.py --encoder vitb --input image.jpg --grayscale

# 列出可用模型
python scripts/infer.py --list-models
```

## 项目结构

```
dpav1-pytorch/
├── depth_anything/
│   ├── models/
│   │   ├── blocks.py          # 基础模块 (ResidualConvUnit, FeatureFusionBlock)
│   │   ├── encoder.py         # DINOv2 编码器
│   │   ├── decoder.py         # DPT 解码器
│   │   └── depth_anything.py  # 完整模型
│   └── utils/
│       └── transforms.py      # 预处理
├── scripts/
│   └── infer.py               # 推理脚本
├── requirements.txt
└── setup.py
```

## 参考

- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) - 原始实现
- [DINOv2](https://github.com/facebookresearch/dinov2) - 编码器
- [DPT](https://github.com/isl-org/DPT) - 解码器架构

## 许可证

GPL-3.0
