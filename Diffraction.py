import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2
from PIL import Image


# 角谱衍射计算
def fresnel_diffraction(image, z, wavelength):
    k = 2 * np.pi / wavelength  # 波数
    height, width = image.shape
    fx = np.fft.fftfreq(width) * width  # x频率分量
    fy = np.fft.fftfreq(height) * height  # y频率分量
    H = np.exp(-1j * k * z) * np.exp(-1j * np.pi * wavelength * z * (fx[:, None] ** 2 + fy[None, :] ** 2))

    # 计算傅里叶变换
    spectrum = fft2(image)

    # 添加随机相位
    random_phase = np.random.rand(height, width) * (2) * np.pi  # 生成[0, 2π)之间的随机相位
    spectrum *= np.exp(1j * random_phase)  # 应用随机相位

    
    U_out = ifft2(spectrum * H)  # 计算逆傅里叶变换
    return np.abs(U_out)  # 返回绝对值（幅度）

# 设置衍射参数


z = 0.1  # 衍射距离，单位可以是米
wavelength = 632.8e-9  # 波长，单位为米

# 输入和输出文件夹路径
input_dir = 'test/test'  # 原始图像文件夹路径
output_dir = 'test/test360'  # 衍射图像保存文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取文件夹中的所有图像
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()  # 确保文件顺序

for image_name in image_files:
    image_path = os.path.join(input_dir, image_name)
    image = np.array(Image.open(image_path).convert('L')) / 255.0  # 转换为灰度图并归一化

    # 计算衍射图
    diffraction_image = fresnel_diffraction(image, z, wavelength)

    # 保存衍射图
    output_path = os.path.join(output_dir, image_name)
    plt.imsave(output_path, diffraction_image, cmap='hot', vmin=0, vmax=1)

    
print("图片衍射完成")
