"""
图像预处理

用于深度估计的图像预处理管道
"""

import numpy as np
import torch
import cv2
from typing import Union, Tuple, Optional


class Resize:
    """
    图像缩放

    将图像缩放到指定尺寸，同时保持宽高比
    """

    def __init__(self, width: int, height: int, keep_aspect_ratio: bool = True,
                 ensure_multiple_of: int = 14, resize_method: str = 'lower_bound'):
        """
        Args:
            width: 目标宽度
            height: 目标高度
            keep_aspect_ratio: 是否保持宽高比
            ensure_multiple_of: 确保尺寸是该值的倍数 (DINOv2 需要 14 的倍数)
            resize_method: 缩放方法
                - 'lower_bound': 保证最小边不小于目标
                - 'upper_bound': 保证最大边不大于目标
                - 'minimal': 最小化缩放
        """
        self.width = width
        self.height = height
        self.keep_aspect_ratio = keep_aspect_ratio
        self.ensure_multiple_of = ensure_multiple_of
        self.resize_method = resize_method

    def _constrain_to_multiple_of(self, val: float) -> int:
        """约束值为 ensure_multiple_of 的倍数"""
        return int(round(val / self.ensure_multiple_of) * self.ensure_multiple_of)

    def _get_size(self, width: int, height: int) -> Tuple[int, int]:
        """计算目标尺寸"""
        scale_width = self.width / width
        scale_height = self.height / height

        if self.keep_aspect_ratio:
            if self.resize_method == 'lower_bound':
                # 保证缩放后的尺寸 >= 目标尺寸
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height

            elif self.resize_method == 'upper_bound':
                # 保证缩放后的尺寸 <= 目标尺寸
                if scale_width < scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height

            elif self.resize_method == 'minimal':
                # 最小化缩放
                if abs(1 - scale_width) < abs(1 - scale_height):
                    scale_height = scale_width
                else:
                    scale_width = scale_height

        new_width = self._constrain_to_multiple_of(scale_width * width)
        new_height = self._constrain_to_multiple_of(scale_height * height)

        return new_width, new_height

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        应用缩放

        Args:
            image: 输入图像 [H, W, C]

        Returns:
            缩放后的图像
        """
        height, width = image.shape[:2]
        new_width, new_height = self._get_size(width, height)

        resized = cv2.resize(
            image, (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        return resized


class NormalizeImage:
    """
    图像归一化

    使用 ImageNet 统计量进行归一化
    """

    def __init__(self, mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225)):
        """
        Args:
            mean: 各通道均值
            std: 各通道标准差
        """
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        应用归一化

        Args:
            image: 输入图像 [H, W, C], 值范围 [0, 1]

        Returns:
            归一化后的图像
        """
        return (image - self.mean) / self.std


class PrepareForNet:
    """
    为网络准备输入

    将图像从 HWC 格式转换为 CHW 格式
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        准备网络输入

        Args:
            image: 输入图像 [H, W, C]

        Returns:
            转换后的图像 [C, H, W]
        """
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = np.ascontiguousarray(image).astype(np.float32)
        return image


class DepthTransform:
    """
    深度估计预处理管道

    完整的预处理流程:
    1. BGR -> RGB
    2. 归一化到 [0, 1]
    3. Resize (保持宽高比，确保 14 的倍数)
    4. ImageNet 归一化
    5. HWC -> CHW

    使用示例:
        transform = DepthTransform(size=518)

        # 处理 numpy 图像
        image = cv2.imread('image.jpg')
        tensor = transform(image)

        # 处理 PIL 图像
        from PIL import Image
        image = Image.open('image.jpg')
        tensor = transform(image)
    """

    def __init__(self, size: int = 518):
        """
        Args:
            size: 目标尺寸 (宽和高相同)
        """
        self.size = size
        self.resize = Resize(
            width=size, height=size,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound'
        )
        self.normalize = NormalizeImage()
        self.prepare = PrepareForNet()

    def __call__(self, image: Union[np.ndarray, 'PIL.Image.Image']) -> torch.Tensor:
        """
        应用预处理

        Args:
            image: 输入图像 (numpy array 或 PIL Image)

        Returns:
            预处理后的张量 [1, 3, H, W]
        """
        # 转换 PIL Image 为 numpy array
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # 如果是 BGR (OpenCV), 转换为 RGB
            if isinstance(image, np.ndarray):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 确保是 float32 且范围 [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # 应用预处理
        image = self.resize(image)
        image = self.normalize(image)
        image = self.prepare(image)

        # 转换为 PyTorch 张量并添加 batch 维度
        tensor = torch.from_numpy(image).unsqueeze(0)

        return tensor

    def postprocess(self, depth: torch.Tensor, original_size: Optional[Tuple[int, int]] = None,
                    colormap: bool = True) -> np.ndarray:
        """
        深度图后处理

        Args:
            depth: 模型输出的深度图 [B, 1, H, W]
            original_size: 原始图像尺寸 (H, W)，用于恢复原始分辨率
            colormap: 是否应用彩色映射

        Returns:
            后处理后的深度图 (numpy array)
        """
        # 移除 batch 维度
        if depth.dim() == 4:
            depth = depth.squeeze(0)
        if depth.dim() == 3:
            depth = depth.squeeze(0)

        # 转换为 numpy
        depth = depth.cpu().numpy()

        # 归一化到 [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 0:
            depth = (depth - depth_min) / (depth_max - depth_min)

        # 恢复原始分辨率
        if original_size is not None:
            depth = cv2.resize(depth, (original_size[1], original_size[0]))

        # 转换为 uint8
        depth = (depth * 255).astype(np.uint8)

        # 应用彩色映射
        if colormap:
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        return depth
