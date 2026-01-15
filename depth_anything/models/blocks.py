"""
基础模块

包含 DPT 解码器所需的核心组件:
- ResidualConvUnit: 残差卷积单元
- FeatureFusionBlock: 特征融合块
"""

import torch
import torch.nn as nn


class ResidualConvUnit(nn.Module):
    """
    残差卷积单元

    结构: 两层 3x3 卷积 + skip 连接
    用于特征精炼和保持空间信息
    """

    def __init__(self, features: int, activation: nn.Module = nn.ReLU(),
                 use_bn: bool = False):
        """
        Args:
            features: 输入/输出通道数
            activation: 激活函数
            use_bn: 是否使用 BatchNorm
        """
        super().__init__()

        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=not use_bn
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=not use_bn
        )

        if use_bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        out = self.activation(x)
        out = self.conv1(out)
        if self.use_bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """
    特征融合块

    功能: 接收多个输入特征，进行融合后上采样
    用于 DPT 解码器中的渐进式特征融合
    """

    def __init__(self, features: int, activation: nn.Module = nn.ReLU(),
                 upsample: bool = True, use_bn: bool = False,
                 align_corners: bool = True):
        """
        Args:
            features: 特征通道数
            activation: 激活函数
            upsample: 是否进行 2x 上采样
            use_bn: 是否使用 BatchNorm
            align_corners: 插值时是否对齐角点
        """
        super().__init__()

        self.upsample = upsample
        self.align_corners = align_corners

        # 两个残差卷积单元
        self.resConfUnit1 = ResidualConvUnit(features, activation, use_bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, use_bn)

        # 1x1 卷积用于特征投影
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 主要输入特征
            skip: 可选的跳跃连接特征

        Returns:
            融合后的特征
        """
        output = x

        # 如果有跳跃连接，先融合
        if skip is not None:
            output = output + self.resConfUnit1(skip)

        output = self.resConfUnit2(output)

        # 上采样 2x
        if self.upsample:
            output = nn.functional.interpolate(
                output, scale_factor=2, mode='bilinear', align_corners=self.align_corners
            )

        output = self.out_conv(output)

        return output


class Interpolate(nn.Module):
    """
    插值模块

    将插值操作封装为可学习模块，方便在 Sequential 中使用
    """

    def __init__(self, scale_factor: float, mode: str = 'bilinear',
                 align_corners: bool = False):
        """
        Args:
            scale_factor: 缩放因子
            mode: 插值模式 ('nearest', 'bilinear', 'bicubic')
            align_corners: 是否对齐角点
        """
        super().__init__()

        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners if mode != 'nearest' else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return nn.functional.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners
        )
