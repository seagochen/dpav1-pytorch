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

    def __init__(self, encoder: str = 'vitb', features: int = 128,
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

            # 转换权重 key 以匹配模型结构
            new_state_dict = cls._convert_state_dict(state_dict)

            # 加载转换后的权重
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

            # 过滤掉预期不匹配的 key (DINOv2 backbone 已由 torch.hub 加载)
            missing = [k for k in missing if not k.startswith('encoder.backbone.')]

            if missing:
                print(f"警告: 以下权重未加载: {missing}")

            print(f"成功加载预训练权重: {repo_id}")

        except Exception as e:
            print(f"警告: 无法加载预训练权重 ({e})")
            print("模型将使用随机初始化的解码器权重")

        return model

    @staticmethod
    def _convert_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        转换预训练权重的 key 以匹配模型结构

        预训练权重结构:
            pretrained.* -> DINOv2 backbone (跳过，使用 torch.hub 加载的)
            depth_head.projects.* -> encoder.projections.* (768 -> [96,192,384,768])
            depth_head.resize_layers.* -> decoder.resize_layers.*
            depth_head.scratch.layer{i}_rn -> decoder.projects.{i-1} (3x3 conv -> 128)
            depth_head.scratch.refinenet{i} -> decoder.fusion_blocks.{i-1}
            depth_head.scratch.output_conv1/2 -> decoder.head.*
        """
        new_state_dict = {}

        for key, value in state_dict.items():
            new_key = None

            # 跳过 DINOv2 backbone 权重 (已由 torch.hub 加载)
            if key.startswith('pretrained.'):
                continue

            # 编码器投影层: depth_head.projects.* -> encoder.projections.*
            if key.startswith('depth_head.projects.'):
                new_key = key.replace('depth_head.projects.', 'encoder.projections.')

            # 解码器调整层: depth_head.resize_layers.* -> decoder.resize_layers.*
            elif key.startswith('depth_head.resize_layers.'):
                new_key = key.replace('depth_head.resize_layers.', 'decoder.resize_layers.')

            # 解码器投影层: depth_head.scratch.layer{i}_rn -> decoder.projects.{i-1}
            # 注意: layer_rn 只有 weight，没有 bias
            elif key.startswith('depth_head.scratch.layer') and '_rn' in key:
                # layer1_rn.weight -> projects.0.weight
                layer_num = int(key.split('.')[2].replace('layer', '').replace('_rn', ''))
                new_key = f'decoder.projects.{layer_num - 1}.weight'

            # 融合块: depth_head.scratch.refinenet{i} -> decoder.fusion_blocks.{i-1}
            elif key.startswith('depth_head.scratch.refinenet'):
                # refinenet1 -> fusion_blocks.0, etc.
                parts = key.split('.')
                refinenet_num = int(parts[2].replace('refinenet', ''))
                rest = '.'.join(parts[3:])
                new_key = f'decoder.fusion_blocks.{refinenet_num - 1}.{rest}'

            # 输出头: depth_head.scratch.output_conv1/2 -> decoder.head.*
            elif key.startswith('depth_head.scratch.output_conv1'):
                # output_conv1 -> head.0
                suffix = key.split('output_conv1.')[-1]
                new_key = f'decoder.head.0.{suffix}'

            elif key.startswith('depth_head.scratch.output_conv2'):
                # output_conv2.0 -> head.2, output_conv2.2 -> head.4
                parts = key.split('.')
                conv_idx = int(parts[3])  # 0 or 2
                suffix = parts[4]  # weight or bias
                # output_conv2.0 -> head.2, output_conv2.2 -> head.4
                head_idx = conv_idx + 2
                new_key = f'decoder.head.{head_idx}.{suffix}'

            if new_key is not None:
                new_state_dict[new_key] = value

        return new_state_dict

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
