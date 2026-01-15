"""
Depth Anything PyTorch 安装脚本
"""

from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='depth_anything',
    version='1.0.0',
    description='Depth Anything: 基于 DINOv2 的单目深度估计模型',
    author='',
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'depth-infer=scripts.infer:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
)
