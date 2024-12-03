import cv2
import numpy as np
import os

dispose_dir = 'seg'  # 数据集文件夹
save_path = 'line'  # 输出文件夹

for i_temp in range(3):
    # 输入模板原图
    img_ori = os.path.join(dispose_dir, str(i_temp + 1) + '.jpg')
    img = cv2.imread(img_ori)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 加一层高斯模糊
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 轮廓
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # 合并所有轮廓
    all_contours = np.vstack(contours)
    # 计算整体的凸包
    overall_hull = cv2.convexHull(all_contours)  # 计算整体凸包
    # 在原图上绘制整体凸包
    cv2.drawContours(img, [overall_hull], -1, (255, 255, 255), 2)  # 绘制整体凸包
    # img=cv2.resize(img,(150,150),interpolation=cv2.INTER_LANCZOS4)
    # 开操作
    # kernel = np.ones((5, 5), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    # 加一层高斯模糊
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # 加一层双边滤波
    img = cv2.bilateralFilter(img, 3, 75, 250)

    cv2.imwrite(os.path.join(save_path, str(i_temp + 1) + '.jpg'), img)

    for j_temp in range(3):
        img_path = os.path.join(dispose_dir, str(i_temp + 1) + '_' + str(j_temp + 1) + '.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_copy = img.copy()
        kernel = np.ones((2, 2), np.uint8)
        img_copy = cv2.dilate(img_copy, kernel, iterations=1)
        # 加一层高斯模糊
        # img = cv2.GaussianBlur(img, (3, 3), 0)
        # 轮廓
        contours, _ = cv2.findContours(img_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # 合并所有轮廓
        all_contours = np.vstack(contours)
        # 计算整体的凸包
        overall_hull = cv2.convexHull(all_contours)  # 计算整体凸包
        # 在原图上绘制整体凸包
        cv2.drawContours(img, [overall_hull], -1, (255, 255, 255), 2)  # 绘制整体凸包
        # img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_LANCZOS4)
        # 开操作 分数没有变化
        # kernel = np.ones((5, 5), np.uint8)
        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        cv2.imwrite(os.path.join(save_path, str(i_temp + 1) + '_' + str(j_temp + 1) + '.jpg'), img)





