import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_small_area(img,a):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    for i in range(1, num_labels):
        if stats[i][4] < a:
            img[labels == i] = 0
    return img

def merge_close_contours(contours, distance_threshold):
    merged_contours = []
    merged_indices = set()  # 用于跟踪已合并的轮廓索引

    for i in range(len(contours)):
        if i in merged_indices:
            continue

        # 当前轮廓的包围盒
        x1, y1, w1, h1 = cv2.boundingRect(contours[i])

        # 初始化合并的轮廓
        merged_contour = contours[i]

        for j in range(i + 1, len(contours)):
            if j in merged_indices:
                continue

            # 获取下一个轮廓的包围盒
            x2, y2, w2, h2 = cv2.boundingRect(contours[j])

            # 计算包围盒之间的中心点距离
            center_distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            if center_distance < distance_threshold:
                merged_contour = np.vstack([merged_contour, contours[j]])  # 垂直堆叠
                merged_indices.add(j)

        # 使用 cv2.convexHull 创建新的轮廓
        merged_contour = cv2.convexHull(merged_contour)
        # 将合并后的轮廓添加到结果中
        merged_contours.append(merged_contour)

    return merged_contours

image_path = 'work/picture1.png'
img = cv2.imread(image_path,0)
img[img<20]=0
#找到所有的轮廓
contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))
# cv2.drawContours(img, contours, -1, (255, 255, 255), -1)
# plt.imshow(img,cmap='gray')
# plt.show()
# #合并距离小于20的轮廓
contours = merge_close_contours(contours, 21)
cv2.drawContours(img, contours, -1, (255, 255, 255), -1)
# plt.imshow(img,cmap='gray')
# plt.show()
# print(len(contours))
#画出前70个最大的轮廓
contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:160]
#在原图上画出所有轮廓
cv2.drawContours(img, contours, -1, (255, 255, 255), -1)
plt.imshow(img,cmap='gray')
plt.show()

#二值化
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.bitwise_not(img, img)
img=remove_small_area(img,5)
plt.imshow(img,cmap='gray')
plt.show()


#膨胀
kernel = np.ones((3, 3), np.uint8) #先膨胀一下，防止把字体没连上腐蚀掉
img = cv2.dilate(img, kernel)
plt.imshow(img,cmap='gray')
plt.show()
#去除小连通域
img=remove_small_area(img,20)
plt.imshow(img,cmap='gray')
plt.show()
# #膨胀
kernel = np.ones((3, 3), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
#去除小连通域
#img=remove_small_area(img,71)
plt.imshow(img,cmap='gray')
plt.show()
# kernel = np.ones((3, 3), np.uint8)
# img = cv2.dilate(img, kernel, iterations=1)
# plt.imshow(img,cmap='gray')
# plt.show()
img1=cv2.imread(image_path,0)
#把img中的白色区域的位置在img1中变成白色
img1[img==255]=220
plt.imshow(img1,cmap='gray')
plt.show()
#otsu二值化
_, img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.bitwise_not(img1, img1)
plt.imshow(img1,cmap='gray')
plt.show()
# _, img1 = cv2.threshold(img1, 215, 255, cv2.THRESH_BINARY )
# cv2.bitwise_not(img1, img1)
#去除小连通域
img1=remove_small_area(img1,120)
cv2.imwrite('picture1_otsu.png',img1)
plt.imshow(img1,cmap='gray')
plt.show()
# #膨胀
# kernel = np.ones((5, 5), np.uint8)
# img1 = cv2.dilate(img1, kernel, iterations=1)
# plt.imshow(img1,cmap='gray')
# plt.show()
# #去除毛刺，开操作
# kernel = np.ones((5, 5), np.uint8)
# img1 = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)
# plt.imshow(img1,cmap='gray')
# plt.show()
#去除毛刺，开操作
kernel = np.ones((5, 5), np.uint8)
img1 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
plt.imshow(img1,cmap='gray')
plt.show()


#cv2.imwrite('picture1.png',img1)

# kernel = np.ones((3, 3), np.uint8)
# img1 = cv2.dilate(img1, kernel, iterations=1)
# #先上一层边缘检测
# img1 = cv2.Canny(img1, 50, 150)
# plt.imshow(img1,cmap='gray')
# plt.show()
#直线检测
lines = cv2.HoughLinesP(img1, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)

# break_points = []
# if lines is not None:
#     for i, line1 in enumerate(lines):
#         x1, y1, x2, y2 = line1[0]
#         for j, line2 in enumerate(lines):
#             if i >= j:
#                 continue
#             x3, y3, x4, y4 = line2[0]
#
#             # 检查是否在同一条网格线（基于斜率和截距）
#             if abs((y2 - y1) * (x4 - x3) - (y4 - y3) * (x2 - x1)) < 10:
#                 # 检测是否存在间隙
#                 if max(x1, x2) < min(x3, x4) or max(y1, y2) < min(y3, y4):
#                     break_points.append(((x2, y2), (x3, y3)))
#
# # 可视化断点
# output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# for p1, p2 in break_points:
#     cv2.line(output, p1, p2, (0, 0, 255), 2)
# # 显示结果
# cv2.imshow('Break Points', output)
# cv2.waitKey(0)
#对直线分组，如果两条直线的截距和斜率都相近，则认为是同一组
# 用于保存直线分组的结果
# line_group = []
# # 遍历每一条直线
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     # 计算直线的斜率和截距
#     slope = (y2 - y1) / (x2 - x1)
#     intercept = y1 - slope * x1
#     # 标记当前直线是否已经被分组
#     is_grouped = False
#     # 遍历每一组直线
#     for group in line_group:
#         # 获取组内第一条直线的斜率和截距
#         slope0, intercept0 = group[0]
#         # 如果两条直线的斜率和截距都相近，则认为是同一组
#         if abs(slope - slope0) < 0.1 and abs(intercept - intercept0) < 10:
#             group.append((slope, intercept))
#             is_grouped = True
#             break
#     # 如果当前直线没有被分组，则创建一个新的组
#     if not is_grouped:
#         line_group.append([(slope, intercept)])
# #对于同一组直线，计算其平均斜率和截距
# # 用于保存合并后的直线
# merged_lines = []
# # 遍历每一组直线
# for group in line_group:
#     # 计算平均斜率和截距
#     slope = np.mean([line[0] for line in group])
#     intercept = np.mean([line[1] for line in group])
#     # 计算直线的端点
#     x1 = 0
#     y1 = round(intercept)
#     x2 = img1.shape[1]
#     y2 = int(slope * x2 + intercept)
#     merged_lines.append(((x1, y1, x2, y2), (slope, intercept)))
# #在黑色图像上绘制检测到的直线
# line_image = np.zeros_like(img1)
# for line in merged_lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)  # 用白色绘制直线

line_image = np.zeros_like(img1)
#在黑色图像上绘制检测到的直线
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 1)  # 用白色绘制直线
#lines = cv2.HoughLinesP(line_image, 1, np.pi / 180, 10, minLineLength=80, maxLineGap=40)
# #直线检测
# lines = cv2.HoughLinesP(img1, 1, np.pi / 180, 10, minLineLength=50, maxLineGap=40)
# line_image = np.zeros_like(img1)

# # 二次检测
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)  # 用白色绘制直线

plt.imshow(line_image,cmap='gray')
plt.show()
#cv2.imwrite('picture1_line.png',line_image)
exit(0)











