# Vision Transformer (ViT) 图像分类训练项目

## 项目简介
本项目基于 PyTorch 实现了一个用于图像分类的 Vision Transformer (ViT) 模型。  
该模型通过将图像序列划分为若干小片段（patch），并利用 Transformer 架构进行特征提取和分类，适合处理序列化的图像数据。  

## 代码核心功能
- **数据加载与预处理**  
  使用自定义的 `ImageTxtDataset` 类，从指定的文本文件读取图片路径和标签，加载训练集与验证集。  
  对图片进行缩放、归一化等预处理，将图片尺寸统一为 `(3, 1, 256)`，后续训练时转为 `(3, 256)`。

- **ViT模型结构**  
  - Patch Embedding：将输入序列按 patch 划分并线性映射为固定维度向量。  
  - 位置编码与分类 token：添加可学习的位置编码和分类标记，保留序列位置信息。  
  - Transformer 编码器层：包括多头自注意力机制和前馈神经网络层。  
  - MLP Head：将 Transformer 输出的分类 token 映射到类别概率。

- **训练与评估**  
  - 使用交叉熵损失函数和 Adam 优化器训练模型。  
  - 训练过程中记录训练损失，并定期在验证集上评估模型性能（损失和准确率）。  
  - 利用 TensorBoard 记录训练与测试指标，方便可视化监控。  
  - 每轮训练结束后保存模型权重。

- **设备支持**  
  自动检测 CUDA 环境，优先使用 GPU 训练。

## 环境依赖
- Python 3.6+
- PyTorch
- torchvision
- einops
- tensorboard

## 使用说明
1. 准备数据集：
   - 按照 `train.txt` 和 `val.txt` 文件格式准备数据列表，内容为图片路径及对应标签。
   - 图片存放于指定的文件夹路径。

2. 安装依赖：
   ```bash
   pip install torch torchvision einops tensorboard
