import cv2  # 导入OpenCV库
import numpy as np  # 导入NumPy库
import os  # 导入os库
dispose_dir = 'data/dataset'  # 数据集文件夹
save_path = 'out1'  # 输出文件夹


def high_reserve(img, ksize, sigm):
    img = img * 1.0
    gauss_out = cv2.GaussianBlur(img, (ksize, ksize), sigm)
    img_out = img - gauss_out + 128
    img_out = img_out / 255.0
    # 饱和处理
    mask_1 = img_out < 0
    mask_2 = img_out > 1
    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    return img_out


def usm(img, number):
    blur_img = cv2.GaussianBlur(img, (0, 0), number)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)

    return usm


def Overlay(target, blend):
    mask = blend < 0.5
    img = 2 * target * blend * mask + (1 - mask) * (1 - 2 * (1 - target) * (1 - blend))
    return img

for pic in os.listdir(dispose_dir):
    img = cv2.imread(os.path.join(dispose_dir, pic))
    img_gas = cv2.GaussianBlur(img, (3, 3), 1.5)

    high = high_reserve(img_gas, 5, 3)
    usm1 = usm(high, 5)
    adjusted_image = (Overlay(img_gas / 255, usm1) * 255).astype(np.uint8)

    # 2. 转换到HSV颜色空间
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转换颜色空间

    # 3. 调整亮度
    target_brightness = 255 # 目标亮度值
    h, s, v = cv2.split(hsv_image)  # 分解HSV通道
    current_brightness = np.mean(v)  # 当前平均亮度
    adjustment_factor = target_brightness / current_brightness  # 计算调整因子
    v = np.clip(v * adjustment_factor, 0, 255).astype(np.uint8)  # 调整亮度
    adjusted_image = cv2.merge((h, s, v))  # 重新组合HSV图像

    #4.调整饱和度
    target_saturation = 1 # 目标饱和度值
    h, s, v = cv2.split(adjusted_image)  # 分解HSV通道
    current_saturation = np.mean(s)  # 当前平均饱和度
    adjustment_factor = target_saturation / current_saturation  # 计算调整因子
    s = np.clip(s * adjustment_factor, 0, 255).astype(np.uint8)  # 调整饱和度
    adjusted_image = cv2.merge((h, s, v))  # 重新组合HSV图像
    #
    #提高对比度
    ##clane = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #
    # # 4. 转换回BGR
    # adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_HSV2BGR)  # 转回BGR
    #转化为灰度图
    adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)

    #
    clane = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(8, 8))
    adjusted_image = clane.apply(adjusted_image)
    # # #otsu二值化
    ret, adjusted_image = cv2.threshold(adjusted_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    cv2.imwrite(os.path.join(save_path, pic), adjusted_image)

    #
    # # 5. 保存处理后的图像
    # output_path = 'output_image.jpg'  # 输出路径
    # cv2.imwrite(output_path, adjusted_image)  # 保存