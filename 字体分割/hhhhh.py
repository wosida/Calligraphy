import cv2
import os
import numpy as np
import math
dispose_dir = './data/dataset'  # 数据集文件夹
save_path = 'out'  # 输出文件夹
def remove_red(image):
    # 将图像从BGR颜色空间转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义深红和浅红的HSV颜色范围
    lower_deep_red = np.array([0, 0, 0])
    upper_deep_red = np.array([15, 255, 255])

    lower_light_red = np.array([150, 0, 0])
    upper_light_red = np.array([180, 255, 255])

    # 创建深红和浅红的mask
    mask_deep_red = cv2.inRange(hsv_image, lower_deep_red, upper_deep_red)
    mask_light_red = cv2.inRange(hsv_image, lower_light_red, upper_light_red)

    b,g,r=image[30,30]

    # 将深红和浅红区域涂成30,30
    image[mask_deep_red > 0] = (b,g,r)
    image[mask_light_red > 0] = (b,g,r)





    # 使用形态学操作来减少噪声
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 对红眼区域进行替换
    result = image.copy()
    # result[mask > 0] = [255, 255, 255]

    return result
def Img1(src,area=40):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(src, connectivity=8)
    img = np.zeros((src.shape[0], src.shape[1]), np.uint8)  # 创建个全0的黑背景
    for i in range(1, num_labels):
        mask = labels == i  # 这一步是通过labels确定区域位置，让labels信息赋给mask数组，再用mask数组做img数组的索引
        if stats[i][4] > area:  # 300是面积 可以随便调
            img[mask] = 255
            # 面积大于300的区域涂白留下，小于300的涂0抹去
        else:
            img[mask] = 0

    return img
def Crop(img):
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if i < 10 or j < 10 or i > img.shape[0] - 10 or j > img.shape[1] - 10:
                img[i][j] = 255
    return img

def extract_red(pic):
    ''''method1：使用inRange方法，拼接mask0,mask1'''

    img = cv2.imdecode(np.fromfile(pic, dtype=np.uint8), -1)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rows, cols, channels = img.shape
    # 区间1
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    # 区间2
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    # 拼接两个区间
    mask = mask0 + mask1
    # 保存图片
    cv2.imencode('.png', mask)[1].tofile(pic)
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

import cv2
import numpy as np




for pic in os.listdir(dispose_dir):

    # #读取图像转化为hsv
    # img = cv2.imread(os.path.join(dispose_dir, pic))
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #
    # #将近似红色的点去掉
    #
    #
    # # thresh=[190,170,190,145,185,180,165,165,200,180,200,170,170,170,180,190,135,175,160,180,170,170,190,170,170]#len
    # # num=0
    #
    # # # 读取图片并二值化，并将图片resize为150
    # # img=cv2.imread(os.path.join(dispose_dir, pic), cv2.IMREAD_GRAYSCALE)
    # # img = cv2.GaussianBlur(img, (3, 3), 0)
    # # # clahe
    # # clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(10, 10))
    # # img = clahe.apply(img)
    #
    #
    # # otsu阈值分割
    # ret, img = cv2.threshold(hsv[:,:,2], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # #去边框
    # for i in range(0, img.shape[0]):
    #     for j in range(0, img.shape[1]):
    #         if i < 10 or j < 10 or i > img.shape[0] - 10 or j > img.shape[1] - 10:
    #             img[i][j] = 255
    # img = cv2.bitwise_not(img)
    # img=Img1(img,30)
    # img = cv2.bitwise_not(img)
    # kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))  # 小的kernel用于腐蚀
    # kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))  # 大的kernel用于膨胀
    # kernel_dilate1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    # kernel_erode1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    # img = cv2.dilate(img, kernel_dilate, iterations=1)  # 膨胀即腐蚀,在(5,5)的范围只要有一个像素为1，那么这个范围内的所有像素都为1
    #
    #
    # img = cv2.erode(img,kernel_erode,iterations=1)
    # img = cv2.bitwise_not(img)
    # img = Img1(img, 30)
    # img = cv2.bitwise_not(img)
    # # img = cv2.dilate(img, kernel_dilate1, iterations=1)
    # # img = cv2.erode(img,kernel_erode1,iterations=1)
    #
    #
    # cv2.imwrite(os.path.join(save_path, pic), img)  # 保存提交图片
    #
    # # #把img中为0的点提取出来
    # # img=cv2.bitwise_not(img)
    # # cv= cv2.findNonZero(img)
    # #
    # #
    # # # 计算这些点的重心
    # # M = cv2.moments(cv)
    # # cX = int(M["m10"] / M["m00"])
    # # cY = int(M["m01"] / M["m00"])
    # # # 找到edge中距离重心最远的点
    # # maxD = 0
    # # maxP = (0, 0)
    # # for i in range(len(cv)):
    # #     d = math.sqrt((cv[i][0][0] - cX) ** 2 + (cv[i][0][1] - cY) ** 2)
    # #     if d > maxD:
    # #         maxD = d
    # #         maxP = cv[i][0]
    # #     # if i==2:
    # #     #     print(maxP,cX,cY,maxD)
    # #
    # # img=cv2.bitwise_not(img)
    # # # 以重心为圆心，最远点为半径画圆
    # # cv2.circle(img, (cX, cY), int(maxD), (0, 0, 0), 2)
    # # cv2.circle(img, maxP, 1, (0, 0, 255), -1)
    # # cv2.circle(img, (cX, cY), 1, (0, 0, 255), -1)


    #边缘检测
    img = cv2.imread(os.path.join(dispose_dir, pic))
    img_gas = cv2.GaussianBlur(img, (3, 3), 1.5)

    high = high_reserve(img_gas, 5, 3)
    usm1 = usm(high, 5)
    adjusted_image = (Overlay(img_gas / 255, usm1) * 255).astype(np.uint8)

    # 2. 转换到HSV颜色空间
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转换颜色空间

    # 3. 调整亮度
    target_brightness = 255  # 目标亮度值
    h, s, v = cv2.split(hsv_image)  # 分解HSV通道
    current_brightness = np.mean(v)  # 当前平均亮度
    adjustment_factor = target_brightness / current_brightness  # 计算调整因子
    v = np.clip(v * adjustment_factor, 0, 255).astype(np.uint8)  # 调整亮度
    adjusted_hsv_image = cv2.merge((h, s, v))  # 重新组合HSV图像

    # 4.调整饱和度
    target_saturation = 1  # 目标饱和度值
    h, s, v = cv2.split(adjusted_hsv_image)  # 分解HSV通道
    current_saturation = np.mean(s)  # 当前平均饱和度
    adjustment_factor = target_saturation / current_saturation  # 计算调整因子
    s = np.clip(s * adjustment_factor, 0, 255).astype(np.uint8)  # 调整饱和度
    adjusted_hsv_image = cv2.merge((h, s, v))  # 重新组合HSV图像
    # # 4. 转换回BGR
    # adjusted_image = cv2.cvtColor(adjusted_hsv_image, cv2.COLOR_HSV2BGR)  # 转回BGR
    # # 转化为灰度图
    # adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
    # clane = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    # adjusted_image = clane.apply(adjusted_image)
    # otsu二值化
    ret, adjusted_image = cv2.threshold(adjusted_image[:,:,2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adjusted_image= cv2.bitwise_not(adjusted_image)
    adjusted_image = Img1(adjusted_image, 30)
    adjusted_image = cv2.bitwise_not(adjusted_image)
    adjusted_image=Crop(adjusted_image)
    # kernel_erode1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 小的kernel用于腐蚀
    # kernel_dilate1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 大的kernel用于膨胀
    # adjusted_image = cv2.dilate(adjusted_image, kernel_dilate1, iterations=1)
    # adjusted_image = cv2.erode(adjusted_image, kernel_erode1, iterations=1)
    kernel=np.ones((3,3),np.uint8)
    #闭操作
    adjusted_image=cv2.morphologyEx(adjusted_image, cv2.MORPH_CLOSE, kernel)
    adjusted_image= cv2.bitwise_not(adjusted_image)
    adjusted_image = Img1(adjusted_image, 40)
    adjusted_image = cv2.bitwise_not(adjusted_image)
    # kernel_erode1 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))  # 小的kernel用于腐蚀
    # kernel_dilate1 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))  # 大的kernel用于膨胀
    # kernel=np.ones((3,3),np.uint8)
    #闭操作
    adjusted_image=cv2.morphologyEx(adjusted_image, cv2.MORPH_CLOSE, kernel)
    # adjusted_image = cv2.dilate(adjusted_image, kernel_dilate1, iterations=1)
    # adjusted_image = cv2.erode(adjusted_image, kernel_erode1, iterations=1)
    #平滑
    #adjusted_image=cv2.blur(adjusted_image,(3,3),0)
    # clane = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    # adjusted_image = clane.apply(adjusted_image)
    img=adjusted_image
    cv2.imwrite(os.path.join(save_path, pic), adjusted_image)



    # # otsu阈值分割
    # ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    #
    # img = Crop(img)
    # img = cv2.bitwise_not(img)
    # img = Img1(img)
    # img = cv2.bitwise_not(img)
    # kernel_erode1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 小的kernel用于腐蚀
    # kernel_dilate1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 大的kernel用于膨胀
    # img = cv2.dilate(img, kernel_dilate1, iterations=1)
    # img = cv2.erode(img, kernel_erode1, iterations=1)
    # cv2.imwrite(os.path.join(save_path, pic), img)
    #
    #
    edge = cv2.Canny(img, 50, 150)
    #提取edge中不为0的点
    edge = cv2.findNonZero(edge)
    #计算这些点的凸包
    hull = cv2.convexHull(edge)
    #画出凸包最小外接圆
    (x, y), radius = cv2.minEnclosingCircle(hull)
    center = (int(x), int(y))
    radius = int(radius)
    img = cv2.imread(os.path.join(save_path, pic))
    cv2.circle(img, center, radius, (0, 0, 0), 2)
    cv2.imwrite(os.path.join(save_path, pic.split('.png')[0] + '-h.png'), img)  # 保存提交图片
    img = cv2.imread(os.path.join(save_path, pic))
    cv2.circle(img, center, radius, (0, 0, 0), -1)
    cv2.imwrite(os.path.join(save_path, pic.split('.png')[0] + '-c.png'), img)  # 保存提交图片


    img=cv2.bitwise_not(img)


    M=cv2.moments(img)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    maxd=0
    maxP=(0,0)
    #找到img中距离重心最远的点
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j]==255:
                d = math.sqrt((i - cX) ** 2 + (j - cY) ** 2)
                if d > maxd:
                    maxd = d
                    maxP = (i,j)
    print(maxP,cX,cY,maxd)



    img=cv2.bitwise_not(img)
    cv2.circle(img, (cX, cY), int(maxd)+4, (0, 0, 0), -1)



    cv2.imwrite(os.path.join(save_path, pic.split('.png')[0]+'-c.png'), img)  # 保存提交图片
    # num+=1