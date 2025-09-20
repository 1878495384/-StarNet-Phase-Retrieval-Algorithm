# # 单张图片测试
# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import os
# import time
# from model import StarNet  # 导入自定义的StarNet模型
# from skimage.metrics import structural_similarity as ssim  # 从skimage库导入结构相似性(SSIM)计算函数

# # 设置设备为GPU（如果可用）或CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = StarNet(base_dim=32).to(device)  # 实例化模型并将其移动到设备上
# model.load_state_dict(torch.load('best_model_starnet180.pth'))  # 加载预训练的模型权重
# model.eval()  # 将模型设置为评估模式


# # 定义PSNR计算函数
# def psnr(target, ref):
#     """计算PSNR"""
#     mse = np.mean((target - ref) ** 2)
#     if mse == 0:  # 如果均方误差为0，避免除以0错误
#         return float('inf')
#     return 20 * np.log10(np.max(ref) / np.sqrt(mse))  # 计算PSNR值


# # 定义SSIM计算函数
# def calculate_ssim(target, ref):
#     """Calculate SSIM"""
#     return ssim(target, ref, data_range=ref.max() - ref.min())  # 计算SSIM值


# # 定义测试模型的函数
# def test_model(model, diffraction_image_path, original_image_path):
#     start_time = time.time()  # 记录开始时间

#     # 加载衍射图像和原始图像
#     diffraction_image = np.array(Image.open(diffraction_image_path).convert('L').resize((256, 256))) / 255.0
#     origin_image = np.array(Image.open(original_image_path).convert('L').resize((256, 256))) / 255.0
    
#     # 将衍射图像转换为张量并添加批次维度
#     diffraction_tensor = torch.tensor(diffraction_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
#     # 通过模型进行预测
#     with torch.no_grad():
#         recovered_image = model(diffraction_tensor)

#     # 将恢复的图像张量转换回numpy数组
#     recovered_image = recovered_image.squeeze().cpu().numpy()
#     end_time = time.time()  # 记录结束时间
#     # 计算PSNR
#     psnr_value = psnr(origin_image, recovered_image)

#     # 计算SSIM
#     ssim_value = calculate_ssim(origin_image, recovered_image)

#     # 绘制原始图像、衍射图像和恢复的图像
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 3, 1)
#     plt.title("Ground Truth Image")
#     plt.imshow(origin_image, cmap='hot')
#     plt.axis('off')

#     plt.subplot(1, 3, 2)
#     plt.title("Diffraction Image")
#     plt.imshow(diffraction_image, cmap='hot')
#     plt.axis('off')

#     plt.subplot(1, 3, 3)
#     plt.title(f"Restored Image")
#     plt.imshow(recovered_image, cmap='hot')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()
   
#     # # 保存恢复的图像
#     # recovered_image_path = 'recovered_image_startnet180.png'  # 设置保存路径和文件名
#     # plt.imsave(recovered_image_path, recovered_image, cmap='gray')  # 使用plt保存图像
    
   
#     print(f"Image generation time: {end_time - start_time:.2f} seconds")  # 打印生成时间
#     print(f"SSIM: {ssim_value:.4f}")  # 打印SSIM值
#     print(f"PSNR: {psnr_value:.4f} dB")  # 打印PSNR值

# # 测试模型
# original_image_path = 'test/w0_4.55_m1_n1.png'  # 原图的文件路径
# diffraction_image_path = 'test/w0_4.55_m1_n1-starnet180.png'  # 替换为你的衍射图像路径
# test_model(model, diffraction_image_path, original_image_path)








# 数据集测试
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
from model import StarNet  # 导入自定义的StarNet模型
from skimage.metrics import structural_similarity as ssim  # 从skimage库导入结构相似性(SSIM)计算函数
from thop import profile  # 导入thop库用于计算FLOPs

# 设置设备为GPU（如果可用）或CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StarNet(base_dim=32).to(device)  # 实例化模型并将其移动到设备上
model.load_state_dict(torch.load('best_model_starnet270.pth'))  # 加载预训练的模型权重
model.eval()  # 将模型设置为评估模式


# 定义PSNR计算函数
def psnr(target, ref):
    """计算PSNR"""
    mse = np.mean((target - ref) ** 2)
    if mse == 0:  # 如果均方误差为0，避免除以0错误
        return float('inf')
    return 20 * np.log10(np.max(ref) / np.sqrt(mse))  # 计算PSNR值


# 定义SSIM计算函数
def calculate_ssim(target, ref):
    """Calculate SSIM"""
    return ssim(target, ref, data_range=ref.max() - ref.min())  # 计算SSIM值


# 定义MSE计算函数
def mse(target, ref):
    """计算MSE"""
    return np.mean((target - ref) ** 2)


# 定义测试模型的函数
def test_model(model, diffraction_image_path, original_image_path, output_folder):
    

    # 加载衍射图像和原始图像
    diffraction_image = np.array(Image.open(diffraction_image_path).convert('L').resize((256, 256))) / 255.0
    origin_image = np.array(Image.open(original_image_path).convert('L').resize((256, 256))) / 255.0
    
    # 将衍射图像转换为张量并添加批次维度
    diffraction_tensor = torch.tensor(diffraction_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    start_time = time.time()  # 记录开始时间
    # 通过模型进行预测
    with torch.no_grad():
        recovered_image = model(diffraction_tensor)
    
    # 将恢复的图像张量转换回numpy数组
    recovered_image = recovered_image.squeeze().cpu().numpy()
    end_time = time.time()  # 记录结束时间
    g_time = start_time - end_time
    # 计算PSNR
    psnr_value = psnr(origin_image, recovered_image)

    # 计算SSIM
    ssim_value = calculate_ssim(origin_image, recovered_image)

    # 计算MSE
    mse_value = mse(origin_image, recovered_image)

    # 绘制原始图像、衍射图像和恢复的图像
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Ground Truth Image")
    plt.imshow(origin_image, cmap='hot')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Diffraction Image")
    plt.imshow(diffraction_image, cmap='hot')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Restored Image\nPSNR: {psnr_value:.4f} dB\nSSIM: {ssim_value:.4f}\nMSE: {mse_value:.6f}")
    plt.imshow(recovered_image, cmap='hot')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    
    # 保存恢复的图像
    file_name = os.path.basename(original_image_path)
    recovered_image_path = os.path.join(output_folder, f"recovered_{file_name}")  # 设置保存路径和文件名
    plt.imsave(recovered_image_path, recovered_image, cmap='hot')  # 使用plt保存图像

    
    print(f"Image generation time: {end_time - start_time:.2f} seconds")  # 打印生成时间

    # 返回PSNR、SSIM和MSE值
    return psnr_value, ssim_value, mse_value, g_time


# 处理文件夹中的所有图像
def process_folder(model, original_folder, diffraction_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    psnr_values = []  # 用于存储所有图像的PSNR值
    ssim_values = []  # 用于存储所有图像的SSIM值
    mse_values = []   # 用于存储所有图像的MSE值
    generation_times = []  # 用于存储所有图像的生成时间
    flops_values = []  # 用于存储所有图像的FLOPs值

    # 遍历原图文件夹中的所有文件
    for file_name in os.listdir(original_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            original_image_path = os.path.join(original_folder, file_name)
            # 假设衍射图文件名与原图文件名相同
            diffraction_image_name = file_name
            diffraction_image_path = os.path.join(diffraction_folder, diffraction_image_name)

            if os.path.exists(diffraction_image_path):
                
                # 加载衍射图像并转换为张量
                diffraction_image = np.array(Image.open(diffraction_image_path).convert('L').resize((256, 256))) / 255.0
                diffraction_tensor = torch.tensor(diffraction_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                # 计算FLOPs
                macs, params = profile(model, inputs=(diffraction_tensor, ), verbose=False)
                flops_values.append(macs)  # MACs (Multiply-Accumulate Operations) 是FLOPs的一种表示

                # 运行模型并获取结果
                psnr_value, ssim_value, mse_value, g_time = test_model(model, diffraction_image_path, original_image_path, output_folder)
                  
                generation_times.append(g_time)# 计算生成时间

                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)
                mse_values.append(mse_value)

    # 计算平均PSNR、SSIM、MSE和FLOPs
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_mse = np.mean(mse_values)
    avg_flops = np.mean(flops_values)  # 平均FLOPs

    # 计算平均图片生成时间
    avg_generation_time = np.mean(generation_times)

    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average FLOPs: {avg_flops:.2f}")
    print(f"Average Image Generation Time: {avg_generation_time:.4f} seconds")


# 测试模型
original_folder = 'test/test'  # 原图的文件夹路径
diffraction_folder = 'test/test270'  # 衍射图的文件夹路径
output_folder = 'recovered_images_startnet(270)'  # 恢复图像的输出文件夹路径
process_folder(model, original_folder, diffraction_folder, output_folder)





# # 打印每一张图片的恢复时间，并且计算平均生成时间
# import torch
# import numpy as np
# from PIL import Image
# import os
# import time
# from model import StarNet  # 导入自定义的StarNet模型

# # 设置设备为GPU（如果可用）或CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = StarNet(base_dim=32).to(device)  # 实例化模型并将其移动到设备上
# model.load_state_dict(torch.load('best_model_starnet180.pth'))  # 加载预训练的模型权重
# model.eval()  # 将模型设置为评估模式

# # 定义测试模型的函数
# def test_model(model, diffraction_image_path):
    
#     # 加载衍射图像
#     diffraction_image = np.array(Image.open(diffraction_image_path).convert('L').resize((256, 256))) / 255.0

#     # 将衍射图像转换为张量并添加批次维度
#     diffraction_tensor = torch.tensor(diffraction_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
#     start_time = time.time()  # 记录开始时间
#     # 通过模型进行预测
#     with torch.no_grad():
#         recovered_image = model(diffraction_tensor)
    
#     # 将恢复的图像张量转换回numpy数组
#     recovered_image = recovered_image.squeeze().cpu().numpy()
#     end_time = time.time()  # 记录结束时间

#     # 返回生成时间
#     return end_time - start_time

# # 处理文件夹中的所有图像
# def process_folder(model, diffraction_folder):
#     generation_times = []  # 用于存储所有图像的生成时间

#     # 遍历衍射图文件夹中的所有文件
#     for file_name in os.listdir(diffraction_folder):
#         if file_name.endswith(('.png', '.jpg', '.jpeg')):
#             diffraction_image_path = os.path.join(diffraction_folder, file_name)

#             # 计算生成时间
#             generation_time = test_model(model, diffraction_image_path)
#             generation_times.append(generation_time)
#             print(f"Generation time for {file_name}: {generation_time*1000:.2f} ms")

#     # 计算平均图片生成时间
#     if generation_times:  # 确保列表不为空
#         avg_generation_time = np.mean(generation_times)
#         print(f"Average Image Generation Time: {avg_generation_time*1000:.2f} ms")
#     else:
#         print("No images were processed.")

# # 测试模型
# diffraction_folder = 'test/test180'  # 衍射图的文件夹路径
# process_folder(model, diffraction_folder)


