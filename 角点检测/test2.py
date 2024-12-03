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

image_path = 'work/picture2.png'
img = cv2.imread(image_path,0)
img[img<40]=0
plt.imshow(img,cmap='gray')
plt.show()
#找到所有的轮廓
contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (255, 255, 255), -1)
plt.imshow(img,cmap='gray')
plt.show()
# #合并距离小于20的轮廓
contours = merge_close_contours(contours, 20)
print(len(contours))
#找到前70个最大的轮廓
#contours = sorted(contours, key=cv2.contourArea, reverse=True)[:220]
#在原图上画出所有轮廓
cv2.drawContours(img, contours, -1, (255, 255, 255), -1)
plt.imshow(img,cmap='gray')
plt.show()
#二值化
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.bitwise_not(img, img)
plt.imshow(img,cmap='gray')
plt.show()

#膨胀
kernel = np.ones((3, 3), np.uint8)
img = cv2.dilate(img, kernel)
plt.imshow(img,cmap='gray')
plt.show()
#去除小连通域
img=remove_small_area(img,30)
plt.imshow(img,cmap='gray')
plt.show()

#膨胀
kernel = np.ones((5, 5), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
plt.imshow(img,cmap='gray')
plt.show()
img1=cv2.imread(image_path,0)
#把img中的白色区域的位置在img1中变成白色
img1[img==255]=225
plt.imshow(img1,cmap='gray')
plt.show()
#otsu二值化
_, img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
cv2.bitwise_not(img1, img1)
#去除小连通域
img1=remove_small_area(img1,150)
plt.imshow(img1,cmap='gray')
plt.show()
cv2.imwrite('picture2_1.png',img1)
# kernel = np.ones((3, 3), np.uint8)
# img1 = cv2.dilate(img1, kernel, iterations=1)
#直线检测
lines = cv2.HoughLinesP(img1, 1, np.pi / 180, 10, minLineLength=50, maxLineGap=40)
line_image = np.zeros_like(img1)
# #第一条直线
# x1, y1, x2, y2 = lines[0][0]
# #画出这两个点和直线
# cv2.line(line_image, (x1, y1), (x2, y2), 255, 1)
# cv2.circle(line_image, (x1, y1), 5, 255, -1)
# cv2.circle(line_image, (x2, y2), 5, 255, -1)
# print(x1, y1, x2, y2)
#
# # 计算两线的斜率
# slope1 = abs((y2 - y1) / (x2 - x1 + 1e-6))
# if slope1 > 100:
#     slope1 = 1e6
#     # 计算截距
#     intercept1 = (x2+x1)//2
# else:
#     slope1 = 0
#     # 计算截距
#     intercept1 = (y2+y1)//2
# print(slope1,intercept1)
# #第二条直线
# x1, y1, x2, y2 = lines[1][0]
# #画出这两个点和直线
# cv2.line(line_image, (x1, y1), (x2, y2), 255, 1)
# cv2.circle(line_image, (x1, y1), 5, 255, -1)
# cv2.circle(line_image, (x2, y2), 5, 255, -1)
# print(x1, y1, x2, y2)
# # 计算两线的斜率
# slope1 = abs((y2 - y1) / (x2 - x1 + 1e-6))
# if slope1 > 100:
#     slope1 = 1e6
#     # 计算截距
#     intercept1 = (x2+x1)//2
# else:
#     slope1 = 0
#     # 计算截距
#     intercept1 = (y2+y1)//2
#
# print(slope1,intercept1)
#
# plt.imshow(line_image,cmap='gray')
# plt.show()
# exit(0)


for i in range(len(lines)):
    for j in range(i + 1, len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        x3, y3, x4, y4 = lines[j][0]

        # 计算两线的斜率
        slope1 = abs((y2 - y1) / (x2 - x1 + 1e-6))
        if slope1>100:
            slope1=1e6
            intercept1=(x2+x1)//2
        else:
            slope1=0
            intercept1=(y2+y1)//2
        slope2 = abs((y4 - y3) / (x4 - x3 + 1e-6))
        if slope2>100:
            slope2=1e6
            intercept2=(x4+x3)//2
        else:
            slope2=0
            intercept2=(y4+y3)//2

        # # 计算截距
        # intercept1 = abs(y1 - slope1 * x1)
        # intercept2 = abs(y3 - slope2 * x3)

        # # 判断斜率是否相似
        # if abs(slope1 - slope2) < 0.1:
            # 判断截距是否接近
        if abs(intercept1 - intercept2) < 6:
            # 补全线段：两端点之间画直线
            mid_x1, mid_y1 = (x2 + x3) // 2, (y2 + y3) // 2
            mid_x2, mid_y2 = (x1 + x4) // 2, (y1 + y4) // 2
            #计算两个中点的斜率
            slope1 = abs((mid_y2 - mid_y1) / (mid_x2 - mid_x1 + 1e-6))
            if slope1>0.5 and slope1<100:
                continue
            cv2.line(line_image, (mid_x1, mid_y1), (mid_x2, mid_y2), 255, 1)

            #cv2.line(line_image, (x2, y2), (x3, y3), 255, 1)

plt.imshow(line_image,cmap='gray')
plt.show()
exit(0)


line_image = np.zeros_like(img1)
#在黑色图像上绘制检测到的直线
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)  # 用白色绘制直线
lines = cv2.HoughLinesP(line_image, 1, np.pi / 180, 10, minLineLength=10, maxLineGap=100)
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
#cv2.imwrite('picture2_line.png',line_image)
exit(0)











