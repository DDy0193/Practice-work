# AlexNet CIFAR-10 图像分类项目

这是一个使用 PyTorch 实现的 AlexNet 深度学习模型，在 CIFAR-10 数据集上进行图像分类任务的项目。项目包含了完整的模型训练、测试和可视化流程。

## 项目概述

本项目实现了经典的 AlexNet 卷积神经网络，并针对 CIFAR-10 数据集（32x32 小尺寸图像）进行了适配优化。主要功能包括：

- AlexNet 模型实现
- 数据加载与预处理
- 模型训练与验证
- TensorBoard 训练可视化
- 模型保存与加载
- 卷积和池化操作可视化

## 环境依赖

- Python 3.8+
- PyTorch 1.12+
- Torchvision
- TensorBoard
- PIL (Pillow)
- NumPy

安装依赖：
```bash
pip install torch torchvision tensorboard pillow numpy