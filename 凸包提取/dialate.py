import cv2
import numpy as np

tu='out/1_3.jpg'
# 读取二值化图像
image = cv2.imread(tu, cv2.IMREAD_GRAYSCALE)

# 找到轮廓
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 假设只有一个轮廓，找到最大的一个
largest_contour = max(contours, key=cv2.contourArea)

# 计算凸包
hull = cv2.convexHull(largest_contour)

# 计算质心
M = cv2.moments(hull)
if M["m00"] != 0:
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
else:
    cx, cy = 0, 0

# 放大比例
scale_factor = 0.999

# 扩大每个顶点
expanded_hull = []
for point in hull:
    x, y = point[0]
    # 按质心放大
    new_x = int(cx + (x - cx) * scale_factor)
    new_y = int(cy + (y - cy) * scale_factor)
    expanded_hull.append([new_x, new_y])

expanded_hull = np.array(expanded_hull, dtype=np.int32)

# 创建一个新的空白图像
expanded_image = np.zeros_like(image)

# 绘制扩大的多边形
cv2.fillConvexPoly(expanded_image, expanded_hull, 255)

# 显示结果
# cv2.imshow('Original Image', image)
# cv2.imshow('Expanded Polygon', expanded_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 保存结果
cv2.imwrite(tu, expanded_image)
