import os
import math
import cv2
from PIL import Image
import numpy as np

def remove_red(image):
    # 将图像从BGR颜色空间转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义深红和浅红的HSV颜色范围
    lower_red1 = np.array([0, 43, 46])
    upper_red1 = np.array([10, 255, 255])

    # 区间2
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])

    # 创建深红和浅红的mask
    mask_deep_red = cv2.inRange(hsv_image, lower_red, upper_red)
    mask_light_red = cv2.inRange(hsv_image, lower_red1, upper_red1)



    # 将深红和浅红区域涂成30,30
    image[mask_deep_red > 0] = (255,255,255)
    image[mask_light_red > 0] = (255,255,255)


    # #使用形态学操作来减少噪声
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 对红眼区域进行替换
    result = image.copy()
    # result[mask > 0] = [255, 255, 255]

    return result
def Img1(src):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(src, connectivity=8)
    img = np.zeros((src.shape[0], src.shape[1]), np.uint8)  # 创建个全0的黑背景
    for i in range(1, num_labels):
        mask = labels == i  # 这一步是通过labels确定区域位置，让labels信息赋给mask数组，再用mask数组做img数组的索引
        if stats[i][4] > 40:  # 面积 可以随便调
            img[mask] = 255

        else:
            img[mask] = 0

    return img
def Crop(img):
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if i < 10 or j < 10 or i > img.shape[0] - 10 or j > img.shape[1] - 10:
                img[i][j] = 255
    return img



img=cv2.imread("01.png")
img=remove_red(img)
cv2.imshow("fin",img)
cv2.waitKey(0)
#去除米字格
#边缘检测
img=cv2.imread("01.png")
edges=cv2.Canny(img,100,200)  #所有梯度值高于低阈值的像素点都被认为是潜在的边缘点。第二个阈值参数为高阈值，用于确定哪些潜在的边缘点是真正的边缘。所有梯度值高于高阈值的像素点都被认为是真正的边缘点。同时，所有梯度值低于低阈值的像素点都被认为不是边缘点。像素梯度在 100 到 200 之间的像素只有在它们与强边缘（梯度大于 200 的像素）相连时才会被视为边
cv2.imshow("edges",edges)
cv2.waitKey(0)

# img=cv2.imread("02.png")
#
# img=remove_red(img)
# cv2.imshow("fin",img)
# cv2.waitKey(0)
# img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# clahe=cv2.createCLAHE(clipLimit=2.6,tileGridSize=(10,10))
# img=clahe.apply(img)
# #otsu阈值分割
# ret,img=cv2.threshold(img,0,255,cv2.THRESH_OTSU)
#
# img=Crop(img)
# img=cv2.bitwise_not(img)
# img=Img1(img)
# img=cv2.bitwise_not(img)
# kernel_erode1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 小的kernel用于腐蚀
# kernel_dilate1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 大的kernel用于膨胀
# img = cv2.dilate(img, kernel_dilate1, iterations=1)
# img=cv2.erode(img,kernel_erode1,iterations=1)
# # cv2.imshow("fin1",img)
# # cv2.waitKey(0)







# img=cv2.imread("01.png",0)
# img = cv2.GaussianBlur(img, (3, 3), 0)
#
# # 边缘检测
# edges = cv2.Canny(img, 50, 200, apertureSize=3)
# cv2.imshow("edges", edges)
# cv2.waitKey(0)
# cv2.bitwise_not(edges, edges)
# # #把边缘以及内部填充为黑色,外部为白色
# # cv2.floodFill(edges, None, (0, 0), 0)
# # cv2.floodFill(edges, None, (0, edges.shape[0] - 1), 0)
# # cv2.floodFill(edges, None, (edges.shape[1] - 1, 0), 0)
# # cv2.floodFill(edges, None, (edges.shape[1] - 1, edges.shape[0] - 1), 0)
# # cv2.imshow("edges", edges)
# # cv2.waitKey(0)
#
#
#
#
#
# #clahe
# clahe=cv2.createCLAHE(clipLimit=2.6,tileGridSize=(10,10))
# img=clahe.apply(img)
# cv2.imshow("clahe",img)
# #
#
# #otsu阈值分割
# ret,img=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow("thresh",img)
# cv2.waitKey(0)
#
#
# for i in range(0,img.shape[0]):
#     for j in range(0,img.shape[1]):
#         if i<10 or j<10 or i>img.shape[0]-10 or j>img.shape[1]-10:
#             img[i][j]=255
#
# kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))  # 小的kernel用于腐蚀
# kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 大的kernel用于膨胀
# kernel_erode1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 小的kernel用于腐蚀
# kernel_dilate1 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))  # 大的kernel用于膨胀
# kernel_erode2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 小的kernel用于腐蚀
# kernel_dilate2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 大的kernel用于膨胀
# # img = cv2.dilate(img, kernel_dilate, iterations=1)  # 膨胀即腐蚀,在(5,5)的范围只要有一个像素为1，那么这个范围内的所有像素都为1
# # img=cv2.erode(img,kernel_erode,iterations=1)
# # img = cv2.dilate(img, kernel_dilate1, iterations=1)
# # img=cv2.erode(img,kernel_erode1,iterations=1)
# # img = cv2.dilate(img, kernel_dilate2, iterations=1)
# # img=cv2.erode(img,kernel_erode2,iterations=1)
# #反转图像
# img=cv2.bitwise_not(img)
# # cv2.imshow("dilate",img)
# # cv2.imwrite("01_dilate.png",img)
# # cv2.waitKey(0)
#
#
# def Img1(src):
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(src, connectivity=8)
#     img = np.zeros((src.shape[0], src.shape[1]), np.uint8)  # 创建个全0的黑背景
#     for i in range(1, num_labels):
#         mask = labels == i  # 这一步是通过labels确定区域位置，让labels信息赋给mask数组，再用mask数组做img数组的索引
#         if stats[i][4] > 40:  # 300是面积 可以随便调
#             img[mask] = 255
#             # 面积大于300的区域涂白留下，小于300的涂0抹去
#         else:
#             img[mask] = 0
#
#     return img
# img=Img1(img)
# img=cv2.bitwise_not(img)
# # cv2.imshow("img",img)
# # cv2.imwrite("01_result.png",img)
# # cv2.waitKey(0)
#
# img = cv2.dilate(img, kernel_dilate2, iterations=1)
# # cv2.imshow("dilate",img)
# # cv2.waitKey(0)
# # img=cv2.bitwise_not(img)
# # img=Img1(img)
# # img=cv2.bitwise_not(img)
# # cv2.imshow("img",img)
# # cv2.waitKey(0)
#
# img=cv2.erode(img,kernel_erode1,iterations=1)
# # cv2.imshow("erode",img)
# # cv2.waitKey(0)
#
#
#
#
#
#
#
#
#
#
#
# # #边缘检测
# # edges=cv2.Canny(img,100,200)  #所有梯度值高于低阈值的像素点都被认为是潜在的边缘点。第二个阈值参数为高阈值，用于确定哪些潜在的边缘点是真正的边缘。所有梯度值高于高阈值的像素点都被认为是真正的边缘点。同时，所有梯度值低于低阈值的像素点都被认为不是边缘点。像素梯度在 100 到 200 之间的像素只有在它们与强边缘（梯度大于 200 的像素）相连时才会被视为边
# # cv2.imshow("edges",edges)
# # cv2.waitKey(0)