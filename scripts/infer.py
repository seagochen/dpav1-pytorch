#!/usr/bin/env python3
"""
Depth Anything 推理脚本

支持单张图像和批量图像处理

使用示例:
    # 单张图像
    python scripts/infer.py --encoder vitb --input image.jpg --output output/

    # 批量处理
    python scripts/infer.py --encoder vits --input images/ --output output/

    # 仅保存深度图
    python scripts/infer.py --encoder vitb --input image.jpg --pred-only

    # 灰度输出
    python scripts/infer.py --encoder vitb --input image.jpg --grayscale
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from depth_anything import DepthAnything, DepthTransform, list_available_models


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Depth Anything 深度估计推理',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 模型参数
    parser.add_argument(
        '--encoder', type=str, default='vitb',
        choices=['vits', 'vitb'],
        help='编码器类型 (default: vitb)'
    )
    parser.add_argument(
        '--list-models', action='store_true',
        help='列出可用的模型'
    )

    # 输入输出
    parser.add_argument(
        '--input', '-i', type=str,
        help='输入图像文件或目录'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='output',
        help='输出目录 (default: output)'
    )

    # 输出选项
    parser.add_argument(
        '--pred-only', action='store_true',
        help='仅保存深度图，不保存并排对比图'
    )
    parser.add_argument(
        '--grayscale', action='store_true',
        help='输出灰度深度图 (默认使用 Inferno 彩色映射)'
    )

    # 设备
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='计算设备 (default: cuda if available)'
    )

    return parser.parse_args()


def get_image_files(input_path: str) -> list:
    """获取输入路径下的所有图像文件"""
    input_path = Path(input_path)

    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        files = []
        for ext in image_extensions:
            files.extend(input_path.glob(f'*{ext}'))
            files.extend(input_path.glob(f'*{ext.upper()}'))
        return sorted(files)
    else:
        raise ValueError(f"输入路径不存在: {input_path}")


def process_image(model: DepthAnything, transform: DepthTransform,
                  image_path: Path, output_dir: Path,
                  pred_only: bool = False, grayscale: bool = False,
                  device: str = 'cuda') -> None:
    """处理单张图像"""
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"警告: 无法读取图像 {image_path}")
        return

    original_size = image.shape[:2]  # (H, W)

    # 预处理
    tensor = transform(image).to(device)

    # 推理
    with torch.no_grad():
        depth = model(tensor)

    # 后处理
    depth_map = transform.postprocess(
        depth,
        original_size=original_size,
        colormap=not grayscale
    )

    # 生成输出文件名
    output_name = image_path.stem + '_depth' + image_path.suffix

    if pred_only:
        # 仅保存深度图
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), depth_map)
    else:
        # 保存并排对比图
        # 如果深度图是彩色的，需要将原图也转换为相同通道数
        if len(depth_map.shape) == 3 and depth_map.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 调整原图尺寸以匹配深度图
            image_resized = cv2.resize(image_rgb, (depth_map.shape[1], depth_map.shape[0]))
            combined = np.hstack([image_resized, cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)])
            combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        else:
            # 灰度深度图
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_resized = cv2.resize(image_gray, (depth_map.shape[1], depth_map.shape[0]))
            combined = np.hstack([image_resized, depth_map])

        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), combined)


def main():
    """主函数"""
    args = parse_args()

    # 列出可用模型
    if args.list_models:
        list_available_models()
        return

    # 检查输入
    if args.input is None:
        print("错误: 请指定输入图像或目录 (--input)")
        return

    # 获取图像文件列表
    try:
        image_files = get_image_files(args.input)
    except ValueError as e:
        print(f"错误: {e}")
        return

    if len(image_files) == 0:
        print(f"警告: 在 {args.input} 中没有找到图像文件")
        return

    print(f"找到 {len(image_files)} 张图像")

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"加载模型: {args.encoder}")
    device = args.device
    model = DepthAnything.from_pretrained(args.encoder)
    model = model.to(device)
    model.eval()

    # 创建预处理器
    transform = DepthTransform(size=518)

    # 处理图像
    print(f"开始处理...")
    for image_path in tqdm(image_files, desc='推理进度'):
        process_image(
            model=model,
            transform=transform,
            image_path=image_path,
            output_dir=output_dir,
            pred_only=args.pred_only,
            grayscale=args.grayscale,
            device=device
        )

    print(f"完成! 结果保存在: {output_dir}")


if __name__ == '__main__':
    main()
