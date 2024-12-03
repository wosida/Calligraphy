import cv2
import numpy as np
import os

def remove_red(img):
    # 红色通道分割
    img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    cv2.imshow('img',img)
    cv2.waitKey(0)

    img=img[:,:,2]
    #img=enhance(img)
    # 二值化
    ret,thresh=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #自适应二值化
    #thresh=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return thresh
def remove_red1(img):
    # 转换图像到HSV颜色空间
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义红色的HSV范围
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # 创建掩码，标记图像中处于红色范围内的像素
    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    # 将红色部分置为0
    img_hsv[mask != 0] = 0


    img = img_hsv[:, :, 2]
    # img=enhance(img)
    # 二值化
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


def enhance(img):
    # #高斯模糊
    # img=cv2.GaussianBlur(img,(3,3),0)

    # 增强对比度
    clahe=cv2.createCLAHE(clipLimit=0.1,tileGridSize=(4,4))
    # 加一层双边滤波
    img = cv2.bilateralFilter(img, 5, 200, 75)
    img=clahe.apply(img)
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # img = cv2.filter2D(img, -1, kernel)
    return img

dispose_dir = 'work'  # 数据集文件夹
save_path = 'seg'  # 输出文件夹

for i_temp in range(3):
    # 输入模板原图
    img_ori=os.path.join(dispose_dir,str(i_temp+1)+'.jpg')
    img=cv2.imread(img_ori)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=enhance(img)
    #otsu二值化
    ret,img=cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # 创建结构元素
    kernel = np.ones((5, 5), np.uint8)
    #膨胀
    img=cv2.dilate(img,kernel,iterations=1)
    cv2.imwrite(os.path.join(save_path,str(i_temp+1)+'.jpg'),img)

    for j_temp in range(3):
        # 输入学生模仿的字的图片
        img_path=os.path.join(dispose_dir,str(i_temp+1)+'_'+str(j_temp+1)+'.jpg')
        img=cv2.imread(img_path)
        img=remove_red1(img)
        img=cv2.bitwise_not(img)
        #otsu二值化
        #ret,img=cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #形态学去除噪点
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((2, 2), np.uint8)
        # # 腐蚀
        img = cv2.erode(img, kernel, iterations=1)
        cv2.imwrite(os.path.join(save_path,str(i_temp+1)+'_'+str(j_temp+1)+'.jpg'),img)
