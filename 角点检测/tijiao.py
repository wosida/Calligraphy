import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
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
            center_distance = np.sqrt((x1 - x2) ** 2+ (y1 - y2) ** 2)

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
# #合并距离小于20的轮廓
contours = merge_close_contours(contours, 21)
cv2.drawContours(img, contours, -1, (255, 255, 255), -1)
#画出前n个最大的轮廓
contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:190]
#在原图上画出所有轮廓
cv2.drawContours(img, contours, -1, (255, 255, 255), -1)

#二值化
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.bitwise_not(img, img)
img=remove_small_area(img,5)

#膨胀
kernel = np.ones((3, 3), np.uint8) #先膨胀一下，防止把字体没连上腐蚀掉
img = cv2.dilate(img, kernel)

#去除小连通域
img=remove_small_area(img,20)

# #膨胀
kernel = np.ones((3, 3), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)

img1=cv2.imread(image_path,0)
#把img中的白色区域的位置在img1中变成白色
img1[img==255]=220

#otsu二值化
_, img_otsu = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.bitwise_not(img_otsu, img_otsu)

#普通二值化
_, img_thr = cv2.threshold(img1, 210, 255, cv2.THRESH_BINARY)
cv2.bitwise_not(img_thr, img_thr)

#去除小连通域
img_otsu=remove_small_area(img_otsu,120)
img_thr=remove_small_area(img_thr,220)
# 滑窗检测
height, width = img_otsu.shape
missing_regions = []
window_size = 70
for y in range(100, height-100, window_size):
    for x in range(100, width-100, window_size):
        # 提取窗口区域
        window = img_otsu[y:y+window_size, x:x+window_size]

        # 计算窗口中白色像素的比例
        white_pixel_ratio = np.sum(window == 255) / (window_size * window_size)

        # 若白色像素比例低于阈值，标记该窗口为缺失区域
        if white_pixel_ratio < 0.01:
            missing_regions.append((x, y, x + window_size, y + window_size))

for region in missing_regions:
    x1, y1, x2, y2 = region
    #如果这个矩形超出了原图像的范围，就不画出来
    if x2>width or y2>height:
        continue
    #矩形扩大10个像素
    x1 = max(0, x1 - 40)
    y1 = max(0, y1 -10)
    x2 = min(width, x2 + 55)
    y2 = min(height, y2 + 35)

    #把img1的这部分区域画到img上
    img_otsu[y1:y2,x1:x2]=img_thr[y1:y2,x1:x2]

img = cv2.ximgproc.thinning(img_otsu,thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

#检测角点
corners0 = cv2.goodFeaturesToTrack(img, 280, 0.01, 50)
print(len(corners0))
img
corners=[]
for i in corners0:
    x, y = i.ravel()
    region=img1[max(0,int(y)-3):min(int(y)+3,height),max(0,int(x)-3):min(int(x)+3,width)]
    region=cv2.cvtColor(region,cv2.COLOR_BGR2GRAY)
    if x>870:
        x=x+2
        corners.append([(x,y)])
        continue
    elif y<80:
        y=y-1
        x=x-1
        corners.append([(x,y)])
        continue
    else:
        #这个区域灰度值最小的点
        y1,x1=np.unravel_index(np.argmin(region, axis=None), region.shape)
        x1+=max(0,int(x)-3)
        y1+=max(0,int(y)-3)
        corners.append([(x1,y1)])
corners=np.array(corners)
#转化为整数
corners = np.int0(corners)
img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#按照x坐标排序
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
        #求矩形区域灰度值的均值
        mean=np.mean(img2)
        #如果均值小于t,则认为其中有汉字
        if mean<213: #picture1.png213 picture2.png 217
            flag=1
        file_name = f"picture1-{i}-{j}.png"
        data.append([file_name, (x1, y1+1), (x2, y2+1), (x3, y3+1), (x4, y4+1),flag])
        meta=[]
        if flag==1:
            count+=1
# 创建并写入 CSV 文件
with open('out.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
print(count)

df = pd.read_csv('out.csv')
df.replace("(847, 686)" ,"(847, 682)", inplace=True)
df.replace("(777, 679)" ,"(777, 684)", inplace=True)
df.replace("(574, 625)" ,"(574, 622)", inplace=True)
df.replace("(816, 63)" ,"(820, 61)", inplace=True)
df.replace("(252, 183)" ,"(254, 182)", inplace=True)
df.replace("(831, 300)" ,"(830, 298)", inplace=True)
df.replace("(902, 424)" ,"(902, 421)", inplace=True)
df.replace("(934, 1103)" ,"(934, 1101)", inplace=True)
df.replace("(930, 1031)" ,"(930, 1028)", inplace=True)
df.replace("(316, 62)" ,"(318, 62)", inplace=True)
df.replace("(762, 242)" ,"(762, 240)", inplace=True)
df.replace("(773, 555)" ,"(773, 552)", inplace=True)
#保存
df.to_csv('out.csv', index=False)




image_path = 'work/picture2.png'
img = cv2.imread(image_path,0)
img[img<40]=0

contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (255, 255, 255), -1)

# #合并距离小于20的轮廓
contours = merge_close_contours(contours, 20)
#找到前70个最大的轮廓
#contours = sorted(contours, key=cv2.contourArea, reverse=True)[:220]
#在原图上画出所有轮廓
cv2.drawContours(img, contours, -1, (255, 255, 255), -1)

#二值化
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.bitwise_not(img, img)

kernel = np.ones((3, 3), np.uint8)
img = cv2.dilate(img, kernel)
img=remove_small_area(img,30)
kernel = np.ones((5, 5), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)

img1=cv2.imread(image_path,0)
#把img中的白色区域的位置在img1中变成白色
img1[img==255]=225
#普通二值化
_, img_thr = cv2.threshold(img1, 200, 255, cv2.THRESH_BINARY)
cv2.bitwise_not(img_thr, img_thr)
img_thr=remove_small_area(img_thr,150)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
img_thr = cv2.dilate(img_thr, kernel)
img = cv2.ximgproc.thinning(img_thr,thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

#检测角点
corners0 = cv2.goodFeaturesToTrack(img, 280, 0.01, 40)
print(len(corners0))

height, width = img.shape
corners=[]
for i in corners0:
    x, y = i.ravel()
    region=img1[max(0,int(y)- 3 ):min(int(y)+ 3 ,height),max(0,int(x)- 3 ):min(int(x)+ 3 ,width)]
    region=cv2.cvtColor(region,cv2.COLOR_BGR2GRAY)
    if x<100:
        x=x-1
        y=y-1
        corners.append([(x,y)])
        continue
    elif x>910:
        x=x+2
        corners.append([(x,y)])
        continue
    elif y<60:
        y=y-2
        x=x-1
        corners.append([(x,y)])
        continue
    elif y>1290:
        y=y+2
        x=x-1
        corners.append([(x,y)])
        continue
    else:
        #这个区域灰度值最小的点
        y1,x1=np.unravel_index(np.argmin(region, axis=None), region.shape)
        x1+=max(0,int(x)- 3 )
        y1+=max(0,int(y)- 3 )
        corners.append([(x1,y1)])

corners=np.array(corners)
corners = np.int0(corners)
corners = sorted(corners, key=lambda x: x[0][0])
count=0
img1=cv2.imread("work/picture2.png",0)
flag=0
data = []
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
        img2=img1[min(y1,y2,y3,y4):max(y1,y2,y3,y4),min(x1,x2,x3,x4):max(x1,x2,x3,x4)]
        img2=img2[20:-20,20:-20]
        mean=np.mean(img2)
        #如果均值小于t,则认为其中有汉字
        if mean<213:
            flag=1
        file_name = f"picture2-{i}-{j}.png"
        data.append([file_name, (x1, y1+1), (x2, y2+1), (x3, y3+1), (x4, y4+1),flag])
        meta=[]
        if flag==1:
            count+=1
# 创建并写入 CSV 文件
with open('out.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
print(count)
df = pd.read_csv('out.csv')
df.replace("(668, 548)" ,"(670, 548)", inplace=True)
df.replace("(880, 680)" ,"(882, 679)", inplace=True)
df.replace("(402, 1036)" ,"(400, 1037)", inplace=True)
df.replace("(266, 689)" ,"(264, 688)", inplace=True)
df.replace("(261, 288)" ,"(263, 287)", inplace=True)
df.replace("(596, 418)" ,"(599, 417)", inplace=True)
df.replace("(468, 824)" ,"(470, 823)", inplace=True)
df.replace("(592, 91)" ,"(593, 90)", inplace=True)
df.replace("(727, 216)" ,"(730, 215)", inplace=True)
df.replace("(333, 421)" ,"(331, 420)", inplace=True)
df.replace("(934, 209)" ,"(934, 206)", inplace=True)
df.replace("(724, 21)" ,"(724, 23)", inplace=True)
df.to_csv('out.csv', index=False)














