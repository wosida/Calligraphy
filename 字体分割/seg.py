import math
import os
from PIL import Image

import numpy as np
from scipy.cluster.vq import kmeans
import cv2
from PIL import ImageEnhance

dispose_dir = 'data/dataset'  # 数据集文件夹
save_path = 'out'  # 输出文件夹
try:
 os.mkdir(save_path)
except FileExistsError:
 pass


for pic in os.listdir(dispose_dir):
 # 读取图片并二值化，并将图片resize为150
 # print(os.path.join(dispose_dir, picroot))
 thisPicRoot = os.path.join(dispose_dir, pic)
 try:
     picOri = cv2.resize(cv2.imread(thisPicRoot), (150, 150))
     imgGray = cv2.resize(cv2.imread(thisPicRoot, 0), (150, 150))
 except:
     continue
 thisPicRoot = os.path.join(dispose_dir, pic)
 img = Image.open(thisPicRoot)
 #增强img的对比度，用pillow的ImageEnhance
 # enhancer = ImageEnhance.Contrast(img)
 # img = enhancer.enhance(1.5)




 w, h = img.size
 points = []
 for count, color in img.getcolors(w * h):
     points.append(color)
 fe = np.array(points, dtype=float)  # 聚类需要是Float或者Double
 book = np.array((fe[100], fe[1], fe[8], fe[8]))  # 聚类中心，初始值
 codebook, distortion = kmeans(fe, 3)
 centers = np.array(codebook, dtype=int)  # 变为色彩，还得转为整数



 minRGB = 255 * 3
 judge = [255, 255, 255]
 for i in range(0, 3):
     if centers[i][1] + centers[i][2] + centers[i][0] < minRGB:
         minRGB = centers[i][1] + centers[i][2] + centers[i][0]
         judge = centers[i]
 for dx in range(0, 150):
     for dy in range(0, 150):
         c = picOri[dx, dy]
         dc = c - judge
         if dc[1]+dc[2]+dc[0]< 60:
             imgGray[dx, dy] = 0
         else:
             imgGray[dx, dy] = 255
         if dx < 20 or dx > 130 or dy < 20 or dy > 130:
             imgGray[dx, dy] = 255
             dst = cv2.blur(imgGray, (5, 5))
 for dx in range(0, 150):
     for dy in range(0, 150):
         if dst[dx, dy]<230:
            dst[dx, dy] = 0
         else:
            dst[dx, dy] = 255

 # dst = cv2.dilate(dst, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
 # dst = cv2.erode(dst, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
 mask = dst

 cv2.imwrite(os.path.join(save_path, pic), dst)  # 保存提交图片






 edge = cv2.Canny(mask, 50, 150)
 # 提取edge中不为0的点
 edge = cv2.findNonZero(edge)
#  #计算这些点的重心
#  M = cv2.moments(edge)
#  cX = int(M["m10"] / M["m00"])
#  cY = int(M["m01"] / M["m00"])
# #找到edge中距离重心最远的点
#  maxD = 0
#  for i in range(len(edge)):
#     d = math.sqrt((edge[i][0][0] - cX) ** 2 + (edge[i][0][1] - cY) ** 2)
#     if d > maxD:
#         maxD = d
#         maxP = edge[i][0]
# #以重心为圆心，最远点为半径画圆
#  cv2.circle(mask, (cX, cY), int(maxD), (0, 0, 0), 2)
# # cv2.circle(mask, maxP, 1, (0, 0, 0), -1)
# # cv2.circle(mask, (cX, cY), 1, (0, 0, 0), -1)


 # 计算这些点的凸包
 hull = cv2.convexHull(edge)
 # 画出凸包最小外接圆
 (x, y), radius = cv2.minEnclosingCircle(hull)
 center = (int(x), int(y))
 radius = int(radius)
 # img = cv2.imread(os.path.join(save_path, pic))
 # cv2.circle(img, center, radius, (0, 0, 0), 2)
 # cv2.imwrite(os.path.join(save_path, pic.split('.png')[0] + '-h.png'), img)  # 保存提交图片
 img = cv2.imread(os.path.join(save_path, pic))
 cv2.circle(mask, center, radius, (0, 0, 0), -1)
 #cv2.imwrite(os.path.join(save_path, pic.split('.png')[0] + '-c.png'), img)  # 保存提交图片




 cv2.imwrite(os.path.join(save_path, pic.split('.png')[0] + '-c.png'), mask)  # 保存提交图片

