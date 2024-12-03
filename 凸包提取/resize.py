import cv2
import numpy as np
import os

dispose_dir = 'out'  # 数据集文件夹
save_path = 'out'  # 输出文件夹


for i_temp in range(3):
    # 输入模板原图
    img_ori = os.path.join(dispose_dir, str(i_temp + 1) + '.jpg')
    img = cv2.imread(img_ori)
    #resize
    img=cv2.resize(img,(150,150),interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(os.path.join(save_path, str(i_temp + 1) + '.jpg'), img)

    for j_temp in range(3):
        img_path = os.path.join(dispose_dir, str(i_temp + 1) + '_' + str(j_temp + 1) + '.jpg')
        img = cv2.imread(img_path)
        #resize
        img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_LANCZOS4)


        cv2.imwrite(os.path.join(save_path, str(i_temp + 1) + '_' + str(j_temp + 1) + '.jpg'), img)





