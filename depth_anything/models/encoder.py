"""
DINOv2 编码器

基于 Facebook DINOv2 Vision Transformer 的特征提取器
支持 vits (小型) 和 vitb (中型) 两种变体
"""

import torch
import torch.nn as nn
from typing import List


# 编码器配置
ENCODER_CONFIGS = {
    'vits': {
        'name': 'dinov2_vits14',
        'features': 384,
        'out_channels': [48, 96, 192, 384],
        'intermediate_layers': [2, 5, 8, 11],
    },
    'vitb': {
        'name': 'dinov2_vitb14',
        'features': 768,
        'out_channels': [96, 192, 384, 768],
        'intermediate_layers': [2, 5, 8, 11],
    },
}


class DINOv2Encoder(nn.Module):
    """
    DINOv2 Vision Transformer 编码器

    从 DINOv2 backbone 提取多尺度特征用于密集预测任务
    """

    def __init__(self, encoder_name: str = 'vitb'):
        """
        Args:
            encoder_name: 编码器名称 ('vits' 或 'vitb')
        """
        super().__init__()

        if encoder_name not in ENCODER_CONFIGS:
            raise ValueError(
                f"不支持的编码器: {encoder_name}. "
                f"可选: {list(ENCODER_CONFIGS.keys())}"
            )

        self.config = ENCODER_CONFIGS[encoder_name]
        self.encoder_name = encoder_name

        # 从 torch.hub 加载 DINOv2 backbone
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            self.config['name'],
            pretrained=True
        )

        # 冻结 backbone 参数 (可选，用于微调时)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

    @property
    def features(self) -> int:
        """返回特征维度"""
        return self.config['features']

    @property
    def out_channels(self) -> List[int]:
        """返回各层输出通道数"""
        return self.config['out_channels']

    @property
    def intermediate_layers(self) -> List[int]:
        """返回提取特征的中间层索引"""
        return self.config['intermediate_layers']

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            4个不同层级的特征列表
        """
        # 获取 patch embedding 后的尺寸
        h, w = x.shape[-2] // 14, x.shape[-1] // 14

        # 使用 DINOv2 的 get_intermediate_layers 获取多尺度特征
        features = self.backbone.get_intermediate_layers(
            x,
            n=self.intermediate_layers,
            reshape=True  # 自动 reshape 为 [B, C, H, W]
        )

        return list(features)


def build_encoder(encoder_name: str = 'vitb') -> DINOv2Encoder:
    """
    构建编码器

    Args:
        encoder_name: 编码器名称

    Returns:
        DINOv2Encoder 实例
    """
    return DINOv2Encoder(encoder_name)


def list_available_encoders():
    """列出可用的编码器"""
    print("\n可用的编码器:")
    print("-" * 60)
    for name, config in ENCODER_CONFIGS.items():
        print(f"  {name}:")
        print(f"    - DINOv2 模型: {config['name']}")
        print(f"    - 特征维度: {config['features']}")
        print(f"    - 输出通道: {config['out_channels']}")
    print("-" * 60)
