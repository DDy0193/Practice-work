# HuaQing
1.熟悉Python基础代码
2.## Sentinel-2 多光谱影像 RGB 合成

本示例演示如何使用 Python 从 Sentinel-2 5 波段 TIFF 文件中提取可见光波段（B2 蓝、B3 绿、B4 红），并将其归一化后合成为一张标准 RGB 图像。

### 功能
1. 读取指定路径下的 Sentinel-2 多光谱 TIFF（含 B2、B3、B4、B8、B11 五个波段）。
2. 提取前三个波段（B2、B3、B4）并转换为浮点数组。
3. 将波段值从原始范围（0–10000）缩放至 0–255，并转为 `uint8`。
4. 合成 RGB 图像并保存为 PNG 格式。

### 环境依赖
- Python 3.6+
- numpy
- rasterio
- Pillow (PIL)

可通过以下命令安装：
```bash
pip install numpy rasterio pillow
