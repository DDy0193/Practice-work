# 🔎 PyTorch 图像分类项目示例（激活函数 + 自定义数据集）

本项目包含以下两个核心模块：

1. ✅ 使用 PyTorch 的 `Sigmoid` 激活函数对图像进行处理，并通过 **TensorBoard** 可视化；
2. ✅ 使用自定义数据集类 `ImageTxtDataset` 从 `.txt` 文件中加载图像路径与标签，用于训练或测试神经网络模型。

---

## 🧠  激活函数模块简介

神经网络中的激活函数用于引入非线性能力，使网络能学习复杂映射关系。此项目中使用的是：

- `Sigmoid`：将输入压缩到 (0, 1) 区间，常用于归一化和二分类输出。

### ✅ 激活函数代码片段：

```python
class Chen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        return self.sigmoid(input)
        
### ✅ 数据集片段：     
class ImageTxtDataset(torch.utils.data.Dataset):
    def __init__(self, txt_path, folder_name, transform):
        ...
    def __getitem__(self, i):
        ...
