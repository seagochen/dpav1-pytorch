"""
Depth Anything 模型

完整的深度估计模型，包含 DINOv2 编码器和 DPT 解码器
支持从 HuggingFace Hub 加载预训练权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from .encoder import DINOv2Encoder, ENCODER_CONFIGS
from .decoder import DPTDecoder


# 可用的模型配置
AVAILABLE_MODELS = {
    'vits': {
        'params': '24.8M',
        'desc': 'ViT-Small - 快速推理，适合边缘设备',
        'hub_repo': 'LiheYoung/depth_anything_vits14',
    },
    'vitb': {
        'params': '97.5M',
        'desc': 'ViT-Base - 均衡性能',
        'hub_repo': 'LiheYoung/depth_anything_vitb14',
    },
}


class DepthAnything(nn.Module, PyTorchModelHubMixin):
    """
    Depth Anything 深度估计模型

    架构: DINOv2 编码器 + DPT 解码器

    支持:
    - vits: ViT-Small (24.8M 参数) - 快速推理
    - vitb: ViT-Base (97.5M 参数) - 均衡性能

    使用示例:
        # 方式1: 从 HuggingFace Hub 加载预训练模型
        model = DepthAnything.from_pretrained('vitb')

        # 方式2: 创建新模型
        model = DepthAnything(encoder='vitb')

        # 推理
        depth = model.predict(image)
    """

    def __init__(self, encoder: str = 'vitb', features: int = 256,
                 use_bn: bool = False, **kwargs):
        """
        Args:
            encoder: 编码器类型 ('vits' 或 'vitb')
            features: 解码器中间特征维度
            use_bn: 是否使用 BatchNorm
        """
        super().__init__()

        if encoder not in ENCODER_CONFIGS:
            raise ValueError(
                f"不支持的编码器: {encoder}. "
                f"可选: {list(ENCODER_CONFIGS.keys())}"
            )

        self.encoder_name = encoder
        self.features = features

        # 获取编码器配置
        encoder_config = ENCODER_CONFIGS[encoder]

        # 构建编码器
        self.encoder = DINOv2Encoder(encoder)

        # 构建解码器
        self.decoder = DPTDecoder(
            in_channels=encoder_config['out_channels'],
            features=features,
            use_bn=use_bn
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W], 值范围 [0, 1] 或已归一化

        Returns:
            相对深度图 [B, 1, H, W]
        """
        # 编码器提取多尺度特征
        features = self.encoder(x)

        # 解码器生成深度图
        depth = self.decoder(features)

        return depth

    def predict(self, x: torch.Tensor, return_numpy: bool = False) -> torch.Tensor:
        """
        推理接口

        Args:
            x: 输入图像 [B, 3, H, W]
            return_numpy: 是否返回 numpy 数组

        Returns:
            归一化的深度图 [B, 1, H, W] (值范围 0-1)
        """
        self.eval()
        with torch.no_grad():
            # 前向传播
            depth = self.forward(x)

            # 双线性插值恢复原始分辨率
            if depth.shape[-2:] != x.shape[-2:]:
                depth = F.interpolate(
                    depth, size=x.shape[-2:],
                    mode='bilinear', align_corners=True
                )

            # 归一化到 [0, 1]
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max - depth_min > 0:
                depth = (depth - depth_min) / (depth_max - depth_min)

            if return_numpy:
                depth = depth.cpu().numpy()

            return depth

    @classmethod
    def from_pretrained(cls, encoder: str = 'vitb', **kwargs) -> 'DepthAnything':
        """
        从 HuggingFace Hub 加载预训练模型

        Args:
            encoder: 编码器类型 ('vits' 或 'vitb')
            **kwargs: 传递给模型构造函数的额外参数

        Returns:
            加载了预训练权重的 DepthAnything 模型
        """
        if encoder not in AVAILABLE_MODELS:
            raise ValueError(
                f"不支持的模型: {encoder}. "
                f"可选: {list(AVAILABLE_MODELS.keys())}"
            )

        model_info = AVAILABLE_MODELS[encoder]
        repo_id = model_info['hub_repo']

        # 创建模型实例
        model = cls(encoder=encoder, **kwargs)

        # 从 HuggingFace Hub 下载权重
        try:
            weights_path = hf_hub_download(
                repo_id=repo_id,
                filename='pytorch_model.bin'
            )

            # 加载权重
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            print(f"成功加载预训练权重: {repo_id}")

        except Exception as e:
            print(f"警告: 无法加载预训练权重 ({e})")
            print("模型将使用随机初始化的解码器权重")

        return model

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'encoder': self.encoder_name,
            'features': self.features,
            'params': AVAILABLE_MODELS.get(self.encoder_name, {}).get('params', 'N/A'),
            'desc': AVAILABLE_MODELS.get(self.encoder_name, {}).get('desc', 'N/A'),
        }


def list_available_models():
    """列出可用的预训练模型"""
    print("\n可用的 Depth Anything 模型:")
    print("-" * 60)
    for name, info in AVAILABLE_MODELS.items():
        print(f"  {name}:")
        print(f"    - 参数量: {info['params']}")
        print(f"    - 描述: {info['desc']}")
        print(f"    - HuggingFace: {info['hub_repo']}")
    print("-" * 60)


def build_model(encoder: str = 'vitb', pretrained: bool = True,
                **kwargs) -> DepthAnything:
    """
    构建 Depth Anything 模型

    Args:
        encoder: 编码器类型
        pretrained: 是否加载预训练权重
        **kwargs: 传递给模型构造函数的额外参数

    Returns:
        DepthAnything 模型实例
    """
    if pretrained:
        return DepthAnything.from_pretrained(encoder, **kwargs)
    else:
        return DepthAnything(encoder=encoder, **kwargs)
