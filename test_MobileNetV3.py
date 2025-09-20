# # 数据集测试，初始通道数
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
from MobileNetV3 import MobileNetV3  # 导入自定义的StarNet模型
from skimage.metrics import structural_similarity as ssim  # 从skimage库导入SSIM计算函数
from thop import profile  # 导入thop库用于计算FLOPs
import pandas as pd  # 导入pandas库用于处理Excel文件

# 设置设备为GPU（如果可用）或CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV3().to(device)  # 实例化支持512x512的StarNet模型
model.load_state_dict(torch.load('best_model_MobileNetV3(270).pth'))  # 加载预训练的模型权重
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
    g_time = end_time - start_time  # 修复生成时间计算

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
    recovered_image_path = os.path.join(output_folder, f"recovered_{file_name}")
    plt.imsave(recovered_image_path, recovered_image, cmap='hot')

    print(f"Image generation time: {g_time:.4f} seconds")

    # 返回PSNR、SSIM、MSE和生成时间
    return psnr_value, ssim_value, mse_value, g_time

# 处理文件夹中的所有图像
def process_folder(model, original_folder, diffraction_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    psnr_values = []
    ssim_values = []
    mse_values = []
    generation_times = []
    flops_values = []

    # 初始化Excel文件或加载现有文件
    excel_file = os.path.join(output_folder, 'metrics_MobileNetV3.xlsx')
    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)
    else:
        df = pd.DataFrame(columns=['Image_Name', 'PSNR_dB', 'SSIM', 'MSE', 'Generation_Time_s', 'FLOPs'])

    # 遍历原图文件夹中的所有文件
    for file_name in os.listdir(original_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            original_image_path = os.path.join(original_folder, file_name)
            diffraction_image_name = file_name
            diffraction_image_path = os.path.join(diffraction_folder, diffraction_image_name)

            if os.path.exists(diffraction_image_path):
                diffraction_image = np.array(Image.open(diffraction_image_path).convert('L').resize((256, 256))) / 255.0
                diffraction_tensor = torch.tensor(diffraction_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                # 计算FLOPs
                macs, params = profile(model, inputs=(diffraction_tensor,), verbose=False)
                flops_values.append(macs)

                # 运行模型并获取结果
                psnr_value, ssim_value, mse_value, g_time = test_model(model, diffraction_image_path, original_image_path, output_folder)

                generation_times.append(g_time)
                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)
                mse_values.append(mse_value)

                # 将结果添加到DataFrame
                new_row = pd.DataFrame({
                    'Image_Name': [file_name],
                    'PSNR_dB': [psnr_value],
                    'SSIM': [ssim_value],
                    'MSE': [mse_value],
                    'Generation_Time_s': [g_time],
                    'FLOPs': [macs]
                })
                df = pd.concat([df, new_row], ignore_index=True)

                # 保存到Excel文件
                df.to_excel(excel_file, index=False)
                print(f"Saved metrics for {file_name} to {excel_file}")

    # 计算平均PSNR、SSIM、MSE和FLOPs
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_mse = np.mean(mse_values)
    avg_flops = np.mean(flops_values)
    avg_generation_time = np.mean(generation_times)

    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average FLOPs: {avg_flops:.2f}")
    print(f"Average Image Generation Time: {avg_generation_time:.4f} seconds")

# 测试模型
original_folder = 'test/test'  # 原图的文件夹路径
diffraction_folder = 'test/test270'  # 衍射图的文件夹路径
output_folder = 'recovered_images_MobileNetV3(270)'  # 恢复图像的输出文件夹路径
process_folder(model, original_folder, diffraction_folder, output_folder)








# # 单张静态测试
# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import os
# import time
# from MobileNetV3 import MobileNetV3  # 导入自定义的StarNet模型
# from skimage.metrics import structural_similarity as ssim  # 从skimage库导入结构相似性(SSIM)计算函数

# # 设置设备为GPU（如果可用）或CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MobileNetV3().to(device)  # 实例化支持512x512的StarNet模型
# model.load_state_dict(torch.load('best_model_MobileNetV3(180).pth'))  # 加载预训练的模型权重
# model.eval()  # 将模型设置为评估模式

# # 定义测试模型的函数
# def test_model(model, diffraction_image_path, original_image_path):
#     start_time = time.time()  # 记录开始时间

#     # 加载衍射图像和原始图像
#     diffraction_image = np.array(Image.open(diffraction_image_path).convert('L').resize((256, 256))) / 255.0  # 修改为512x512
#     origin_image = np.array(Image.open(original_image_path).convert('L').resize((256, 256))) / 255.0  # 修改为512x512
    
#     # 将衍射图像转换为张量并添加批次维度
#     diffraction_tensor = torch.tensor(diffraction_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
#     # 通过模型进行预测
#     with torch.no_grad():
#         recovered_image = model(diffraction_tensor)

#     # 将恢复的图像张量转换回numpy数组
#     recovered_image = recovered_image.squeeze().cpu().numpy()
#     end_time = time.time()  # 记录结束时间

#     # 计算SSIM和PSNR
#     data_range = origin_image.max() - origin_image.min()
#     ssim_value = ssim(origin_image, recovered_image, data_range=data_range)
#     mse = np.mean((origin_image - recovered_image) ** 2)
#     psnr_value = 20 * np.log10(data_range / np.sqrt(mse))

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
#     plt.title("Restored Image")
#     plt.imshow(recovered_image, cmap='hot')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()
   
#     print(f"Image generation time: {end_time - start_time:.2f} seconds")  # 打印生成时间
#     print(f"SSIM: {ssim_value:.4f}")  # 打印SSIM值
#     print(f"PSNR: {psnr_value:.4f} dB")  # 打印PSNR值

# # 测试模型
# original_image_path = 'test/w0_9_m3_n2.png'  # 原图的文件路径
# diffraction_image_path = 'test/w0_9_m3_n2-mobilenet270.png'  # 替换为你的衍射图像路径
# test_model(model, diffraction_image_path, original_image_path)











# # 计算fps 动态
# import torch
# import torch.nn as nn
# import numpy as np
# import cv2  # 导入 OpenCV 用于视频处理
# import os
# import datetime
# from MobileNetV2 import MobileNetV2  # 导入自定义的StarNet模型

# # 设置设备为GPU（如果可用）或CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MobileNetV2().to(device)  # 实例化支持512x512的StarNet模型
# model.load_state_dict(torch.load('best_model_MobileNetV2(cat)y.pth'))  # 加载预训练模型权重
# model.eval()  # 设置模型为评估模式

# def process_video(model, input_video_path, output_dir="videos"):
#     """处理输入视频，恢复每帧并保存为新视频，同时实时显示，并计算平均FPS"""
#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)

#     # 打开输入视频
#     cap = cv2.VideoCapture(input_video_path)
#     if not cap.isOpened():
#         print("无法打开视频文件")
#         return

#     # 获取视频属性
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"输入视频分辨率: {width}x{height}, 帧率: {fps}")

#     # 检查分辨率是否为 2048x2048
#     if width != 2048 or height != 2048:
#         print("警告：视频分辨率不是 2048x2048，模型输入将调整为 512x512")

#     # 设置输出视频
#     timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
#     output_video_path = os.path.join(output_dir, f"restored_{timestamp}.avi")
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (512, 512), isColor=False)

#     # 创建显示窗口
#     cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Original Video", 512, 512)
#     cv2.namedWindow("Restored Video", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Restored Video", 512, 512)

#     frame_count = 0
#     total_processing_time = 0.0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("视频读取结束")
#             break

#         # 记录开始时间
#         start_time = cv2.getTickCount()

#         # 转换为灰度并调整大小为 512x512
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         resized_frame = cv2.resize(gray_frame, (512, 512)) / 255.0

#         # 将帧转换为张量并添加批次维度
#         frame_tensor = torch.tensor(resized_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

#         # 通过模型进行预测
#         with torch.no_grad():
#             restored_frame = model(frame_tensor)

#         # 将恢复的帧转换回 NumPy 数组
#         restored_frame = restored_frame.squeeze().cpu().numpy()
#         # 归一化到 [0, 255] 并转换为 uint8
#         restored_frame = (restored_frame * 255).clip(0, 255).astype(np.uint8)

#         # 记录结束时间并计算处理时间
#         end_time = cv2.getTickCount()
#         processing_time = (end_time - start_time) / cv2.getTickFrequency()
#         total_processing_time += processing_time

#         # 保存恢复帧到视频
#         out.write(restored_frame)

#         # 显示原帧和恢复帧
#         cv2.imshow("Original Video", cv2.resize(gray_frame, (512, 512)))
#         cv2.imshow("Restored Video", restored_frame)

#         frame_count += 1
#         print(f"处理帧 {frame_count}, 当前FPS: {1 / processing_time:.2f}")

#         # 按 'q' 键退出
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # 释放资源
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     # 计算并打印平均FPS
#     if frame_count > 0:
#         average_fps = frame_count / total_processing_time
#         print(f"恢复视频已保存到: {output_video_path}")
#         print(f"总帧数: {frame_count}, 总处理时间: {total_processing_time:.2f} 秒")
#         print(f"平均FPS: {average_fps:.2f}")
#     else:
#         print("未处理任何帧")

# def main():
#     """主函数：处理输入视频并恢复"""
#     input_video_path = 'test/cat.avi'  # 替换为你的输入视频路径
#     process_video(model, input_video_path)

# if __name__ == "__main__":
#     main()