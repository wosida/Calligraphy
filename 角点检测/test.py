import cv2
import numpy as np
import matplotlib.pyplot as plt
# img=cv2.imread('picture1_otsu.png',0)
# img1=cv2.imread('picture1.png',0)
#
# plt.imshow(img,cmap='gray')
# plt.show()
# # 滑窗检测
# height, width = img.shape
# missing_regions = []
# window_size = 70
# for y in range(100, height-100, window_size):
#     for x in range(100, width-100, window_size):
#         # 提取窗口区域
#         window = img[y:y+window_size, x:x+window_size]
#
#         # 计算窗口中白色像素的比例
#         white_pixel_ratio = np.sum(window == 255) / (window_size * window_size)
#
#         # 若白色像素比例低于阈值，标记该窗口为缺失区域
#         if white_pixel_ratio < 0.01:
#             missing_regions.append((x, y, x + window_size, y + window_size))
#
# # 可视化结果
# output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# img1=cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
# for region in missing_regions:
#     x1, y1, x2, y2 = region
#     #如果这个矩形超出了原图像的范围，就不画出来
#     if x2>width or y2>height:
#         continue
#     #矩形扩大10个像素
#     x1 = max(0, x1 - 40)
#     y1 = max(0, y1 - 10)
#     x2 = min(width, x2 + 55)
#     y2 = min(height, y2 + 35)
#
#     cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     plt.imshow(img1)
#     plt.show()
#     img1=cv2.imread('picture1.png',0)
#     #把img1的这部分区域画到img上
#     img[y1:y2,x1:x2]=img1[y1:y2,x1:x2]
#
#
# # 显示结果
# plt.figure(figsize=(10, 10))
# plt.imshow(output_image)
# plt.title("Missing Regions Detected by Sliding Window")
# plt.show()
# plt.imshow(img,cmap='gray')
# plt.show()
# cv2.imwrite('picture1_line_1.png',img)

def remove_small_area(img,a):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    for i in range(1, num_labels):
        if stats[i][4] < a:
            img[labels == i] = 0
    return img
img=cv2.imread('picture1_line_1.png',0)
#闭操作
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


plt.imshow(img,cmap='gray')
plt.show()
# #去除小面积区域
img = remove_small_area(img, 30)
plt.imshow(img,cmap='gray')
plt.show()
#提取骨架
img = cv2.ximgproc.thinning(img,thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
plt.imshow(img,cmap='gray')
plt.show()
#检测角点
corners = cv2.goodFeaturesToTrack(img, 280, 0.01, 50)
print(len(corners))
img1=cv2.imread("work/picture1.png")
#画出角点
#img1=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
for i in corners:
    x, y = i.ravel()
    cv2.circle(img1, (int(x), int(y)), 2, (0,0,255), -1)
cv2.imwrite("picture1_corner.png", img1)
plt.imshow(img1)
plt.show()
exit(0)

#
#直线检测
lines = cv2.HoughLinesP(img, 1, np.pi / 180, 30, minLineLength=100, maxLineGap=10)
#直接在原图上绘制直线
line_image = np.zeros_like(img)
for line in lines:
    x1, y1, x2, y2 = line[0]
    slope1 = abs((y2 - y1) / (x2 - x1 + 1e-6))
    if slope1 > 0.1 and slope1 < 1:
        continue
    cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 1)  # 用白色绘制直线
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)  # 用白色绘制直线
plt.imshow(line_image,cmap='gray')
plt.show()


# img=remove_small_area(img,70)
# plt.imshow(img,cmap='gray')
# plt.show()
#kai
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel,iterations=1)
plt.imshow(img,cmap='gray')
plt.show()
#提取骨架
img = cv2.ximgproc.thinning(img,thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
plt.imshow(img,cmap='gray')
plt.show()
#去除毛刺
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
plt.imshow(img,cmap='gray')
plt.show()
cv2.imwrite('picture1_1.png',img)

#角点检测
corners = cv2.goodFeaturesToTrack(img, 280, 0.02, 40)
#画出角点
img=cv2.imread('work/picture1.png')
for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (int(x), int(y)), 3, (0,0,255), -1)
plt.imshow(img)
plt.show()
exit(0)
lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
print(len(lines))
#直接在原图上绘制直线
line_image = np.zeros_like(img)
for line in lines:
    x1, y1, x2, y2 = line[0]
    slope1 = abs((y2 - y1) / (x2 - x1 + 1e-6))
    if slope1 > 0.1 and slope1 < 1:
        continue
    cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 1)  # 用白色绘制直线
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)  # 用白色绘制直线

plt.imshow(line_image,cmap='gray')
plt.show()
