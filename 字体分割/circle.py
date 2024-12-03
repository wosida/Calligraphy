import cv2
import math



img_path='12.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图
img = cv2.bitwise_not(img)
m = cv2.moments(img)
# 计算图像的质心
coi = (m['m10'] / m['m00'], m['m01'] / m['m00'])
maxd = 0
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i, j] == 255:
            d = math.sqrt((i - coi[1]) ** 2 + (j - coi[0]) ** 2)
            if d > maxd:
                maxd = d
img = cv2.bitwise_not(img)
cv2.circle(img, (int(coi[0]), int(coi[1])),(int(maxd)+1), 0, -1)
cv2.imwrite((img_path.split('.png')[0]+'-c.png'),img)  # 保存提交图片