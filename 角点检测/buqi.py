import cv2
import numpy as np
import matplotlib.pyplot as plt
def is_intersecting(x1, y1, x2, y2, x3, y3, x4, y4):
    # 定义一个函数判断点 (px, py) 是否在点 (ax, ay) 和 (bx, by) 之间
    def on_segment(ax, ay, bx, by, px, py):
        return min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by)

    # 计算叉积 (向量 (ax, ay)->(bx, by) 和 (ax, ay)->(cx, cy) 的叉积)
    def cross_product(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    # 计算叉积以判断线段 (x1, y1)-(x2, y2) 和 (x3, y3)-(x4, y4) 是否相交
    d1 = cross_product(x3, y3, x4, y4, x1, y1)
    d2 = cross_product(x3, y3, x4, y4, x2, y2)
    d3 = cross_product(x1, y1, x2, y2, x3, y3)
    d4 = cross_product(x1, y1, x2, y2, x4, y4)

    # 检查是否相交
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True  # 两条线段相交

    # 检查是否共线且有交点
    if d1 == 0 and on_segment(x3, y3, x4, y4, x1, y1):
        return True
    if d2 == 0 and on_segment(x3, y3, x4, y4, x2, y2):
        return True
    if d3 == 0 and on_segment(x1, y1, x2, y2, x3, y3):
        return True
    if d4 == 0 and on_segment(x1, y1, x2, y2, x4, y4):
        return True

    return False  # 不相交


img=cv2.imread("picture2_line.png",0)
#直线检测
lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
line_image = np.zeros_like(img)
#在黑色图像上绘制检测到的直线
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 1)  # 用白色绘制直线
plt.imshow(line_image,cmap='gray')
plt.show()
line_image1 = np.zeros_like(img)
count=0
for i in range(len(lines)):
    for j in range(i + 1, len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        x3, y3, x4, y4 = lines[j][0]
        #判断这两条直线是否有交集，如果有交集则跳过
        if is_intersecting(x1, y1, x2, y2, x3, y3, x4, y4):
            continue


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
        if abs(intercept1 - intercept2) < 3:
            # 补全线段：两端点之间画直线
            mid_x1, mid_y1 = (x2 + x3) // 2, (y2 + y3) // 2
            mid_x2, mid_y2 = (x1 + x4) // 2, (y1 + y4) // 2
            #计算两个中点的斜率
            slope1 = abs((mid_y2 - mid_y1) / (mid_x2 - mid_x1 + 1e-6))
            if slope1>0.1 and slope1<100:
                continue
            cv2.line(line_image1, (mid_x1, mid_y1), (mid_x2, mid_y2), 255, 1)
            # plt.imshow(line_image1, cmap='gray')
            # plt.show()
            count+=1

            #cv2.line(line_image, (x2, y2), (x3, y3), 255, 1)
print(count)
plt.imshow(line_image1,cmap='gray')
plt.show()
#把两个图像叠加
line_image=cv2.addWeighted(line_image,0.5,line_image1,0.5,0)
plt.imshow(line_image,cmap='gray')
plt.show()