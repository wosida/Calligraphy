import cv2
import matplotlib.pyplot as plt
import numpy as np
import csv
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
# lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50)
# line_image = np.zeros_like(img)
# #在黑色图像上绘制检测到的直线
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 1)  # 用白色绘制直线
# plt.imshow(line_image,cmap='gray')
# plt.show()

# #提取骨架
img = cv2.ximgproc.thinning(img,thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
plt.imshow(img,cmap='gray')
plt.show()

#检测角点
corners = cv2.goodFeaturesToTrack(img, 280, 0.01, 50)
print(len(corners))

#去除直线上的毛刺
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# #直线检测
# lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
# line_image = np.zeros_like(img)
# #在黑色图像上绘制检测到的直线
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 1)  # 用白色绘制直线
# plt.imshow(line_image,cmap='gray')
# plt.show()

img1=cv2.imread("work/picture1.png")
#画出角点
#img1=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
for i in corners:
    x, y = i.ravel()
    cv2.circle(img1, (int(x), int(y)), 2, (0,0,255), -1)
cv2.imwrite("picture1_corner.png", img1)
plt.imshow(img1)
plt.show()

#转化为整数
corners = np.int0(corners)
img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#找到横坐标最小的二十个点
corners = sorted(corners, key=lambda x: x[0][0])
count=0
img1=cv2.imread("work/picture1.png",0)
flag=0
data = [
    ["File Name", "Top Left", "Top Right", "Bottom Right", "Bottom Left","Has Text"],
]
for i in range(13):
    i_=i*20
    corners1=corners[i_:i_+20]
    corners2=corners[i_+20:i_+40]
    corners1 = sorted(corners1, key=lambda x: x[0][1])
    corners2 = sorted(corners2, key=lambda x: x[0][1])
    for j in range(19):
        flag=0
        x1,y1=corners1[j][0]
        x2,y2=corners2[j][0]
        x3,y3=corners2[j+1][0]
        x4,y4=corners1[j+1][0]
        #截取img1中在四个点围成的矩形区域
        img2=img1[min(y1,y2,y3,y4):max(y1,y2,y3,y4),min(x1,x2,x3,x4):max(x1,x2,x3,x4)]
        #裁掉上下左右的白边
        img2=img2[20:-20,20:-20]
        # if i==9 and j==11: #picture2的特殊情况
        #     plt.imshow(img2,cmap='gray')
        #     plt.show()

        if i==4 and j==6: #picture1的特殊情况
            plt.imshow(img2,cmap='gray')
            plt.show()
        #求矩形区域灰度值的均值
        mean=np.mean(img2)
        #如果均值小于t,则认为其中有汉字
        if mean<213: #picture1.png213 picture2.png 217
            flag=1
        file_name = f"picture1-{i}-{j}.png"
        data.append([file_name, (x1, y1), (x2, y2), (x3, y3), (x4, y4),flag])
        #print((i,j),flag)
        meta=[]
        if flag==1:
            count+=1
            #画圈
            cv2.circle(img1, (x1, y1), 3, (0, 0, 255), -1)
            cv2.circle(img1, (x2, y2), 3, (0, 0, 255), -1)
# 创建并写入 CSV 文件
with open('out.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
print(count)
plt.imshow(img1,cmap='gray')
plt.show()








