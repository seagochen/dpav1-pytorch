"""
DPT 解码器

Dense Prediction Transformer (DPT) 解码器头
用于将编码器的多尺度特征转换为深度预测
"""

import torch
import torch.nn as nn
from typing import List

from .blocks import FeatureFusionBlock


class DPTDecoder(nn.Module):
    """
    DPT 解码器

    将来自 DINOv2 编码器的多尺度特征融合并生成深度图

    架构:
    1. 投影层 (Project): 将各层特征投影到统一维度
    2. 调整层 (Resize): 调整特征图尺寸
    3. 融合块 (Fusion): 渐进式特征融合
    4. 输出头 (Head): 生成最终深度图
    """

    def __init__(self, in_channels: List[int], features: int = 256,
                 use_bn: bool = False):
        """
        Args:
            in_channels: 各层输入通道数列表 (4个值)
            features: 中间特征通道数
            use_bn: 是否使用 BatchNorm
        """
        super().__init__()

        assert len(in_channels) == 4, "需要4个输入通道数"

        self.features = features

        # 投影层: 将各层特征映射到统一维度
        self.projects = nn.ModuleList([
            nn.Conv2d(in_ch, features, kernel_size=1, stride=1, padding=0)
            for in_ch in in_channels
        ])

        # 调整层: 使用转置卷积或插值调整尺寸
        # 层级关系: layer4 -> layer3 -> layer2 -> layer1 (从深到浅)
        self.resize_layers = nn.ModuleList([
            # layer1: 上采样 4x
            nn.ConvTranspose2d(features, features, kernel_size=4, stride=4, padding=0),
            # layer2: 上采样 2x
            nn.ConvTranspose2d(features, features, kernel_size=2, stride=2, padding=0),
            # layer3: 保持尺寸
            nn.Identity(),
            # layer4: 下采样 2x (stride=2 conv)
            nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1),
        ])

        # 特征融合块: 渐进式融合
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlock(features, nn.ReLU(False), upsample=True, use_bn=use_bn),
            FeatureFusionBlock(features, nn.ReLU(False), upsample=True, use_bn=use_bn),
            FeatureFusionBlock(features, nn.ReLU(False), upsample=True, use_bn=use_bn),
            FeatureFusionBlock(features, nn.ReLU(False), upsample=False, use_bn=use_bn),
        ])

        # 输出头: 生成单通道深度图
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),  # 确保深度为正值
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播

        Args:
            features: 来自编码器的4个特征图列表
                      [layer1, layer2, layer3, layer4]
                      从浅层到深层

        Returns:
            深度图 [B, 1, H, W]
        """
        assert len(features) == 4, f"需要4个特征图，但收到 {len(features)} 个"

        # 1. 投影到统一维度
        projected = [proj(feat) for proj, feat in zip(self.projects, features)]

        # 2. 调整尺寸
        resized = [resize(proj) for resize, proj in zip(self.resize_layers, projected)]

        # 3. 渐进式融合 (从深层到浅层)
        # layer4 开始
        path = resized[3]  # layer4
        path = self.fusion_blocks[3](path)

        # 融合 layer3
        path = self.fusion_blocks[2](path, resized[2])

        # 融合 layer2
        path = self.fusion_blocks[1](path, resized[1])

        # 融合 layer1
        path = self.fusion_blocks[0](path, resized[0])

        # 4. 输出头生成深度图
        depth = self.head(path)

        return depth


def build_decoder(in_channels: List[int], features: int = 256,
                  use_bn: bool = False) -> DPTDecoder:
    """
    构建 DPT 解码器

    Args:
        in_channels: 各层输入通道数
        features: 中间特征维度
        use_bn: 是否使用 BatchNorm

    Returns:
        DPTDecoder 实例
    """
    return DPTDecoder(in_channels, features, use_bn)
