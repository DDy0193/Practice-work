import numpy as np
import rasterio
from PIL import Image
import os

# 输入路径（包含5波段的哨兵2号图像）
input_tif = r"D:\Download\2019_1101_nofire_B2348_B12_10m_roi.tif"
# 输出图像路径
output_image = r"D:\Download\output\rgb_image.png"

def load_sentinel2_rgb_image(tif_path):
    with rasterio.open(tif_path) as src:
        # 读取 5 个波段中的前三个（RGB），假设顺序为 B2, B3, B4, B8, B11
        blue = src.read(1).astype(np.float32)   # B2
        green = src.read(2).astype(np.float32)  # B3
        red = src.read(3).astype(np.float32)    # B4
        # 其他两个波段是 NIR（B8）和 SWIR（B11），我们不处理，只读 RGB

    return red, green, blue

def scale_to_255(band):
    """
    将波段从 0-10000 压缩（归一化）到 0-255 并转为 uint8
    """
    return np.clip((band / 10000) * 255, 0, 255).astype(np.uint8)

def save_rgb_image(red, green, blue, output_path):
    rgb = np.dstack((scale_to_255(red),
                     scale_to_255(green),
                     scale_to_255(blue)))
    img = Image.fromarray(rgb, mode='RGB')
    img.save(output_path)
    print(f"✅ RGB 图像已保存: {output_path}")

if __name__ == "__main__":
    if not os.path.exists(os.path.dirname(output_image)):
        os.makedirs(os.path.dirname(output_image))

    red, green, blue = load_sentinel2_rgb_image(input_tif)
    save_rgb_image(red, green, blue, output_image)
