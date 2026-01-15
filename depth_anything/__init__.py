"""
Depth Anything PyTorch

基于 DINOv2 的单目深度估计模型

使用示例:
    from depth_anything import DepthAnything

    # 加载预训练模型
    model = DepthAnything.from_pretrained('vitb')
    model.eval()

    # 推理
    depth = model.predict(image_tensor)

可用模型:
    - vits: ViT-Small (24.8M 参数) - 快速推理
    - vitb: ViT-Base (97.5M 参数) - 均衡性能
"""

__version__ = '1.0.0'

from .models import (
    # 主模型
    DepthAnything,
    build_model,
    list_available_models,
    AVAILABLE_MODELS,
    # 编码器
    DINOv2Encoder,
    build_encoder,
    list_available_encoders,
    ENCODER_CONFIGS,
    # 解码器
    DPTDecoder,
    build_decoder,
    # 基础模块
    ResidualConvUnit,
    FeatureFusionBlock,
)

from .utils import (
    DepthTransform,
    Resize,
    NormalizeImage,
    PrepareForNet,
)

__all__ = [
    '__version__',
    # 主模型
    'DepthAnything',
    'build_model',
    'list_available_models',
    'AVAILABLE_MODELS',
    # 编码器
    'DINOv2Encoder',
    'build_encoder',
    'list_available_encoders',
    'ENCODER_CONFIGS',
    # 解码器
    'DPTDecoder',
    'build_decoder',
    # 基础模块
    'ResidualConvUnit',
    'FeatureFusionBlock',
    # 工具
    'DepthTransform',
    'Resize',
    'NormalizeImage',
    'PrepareForNet',
]
