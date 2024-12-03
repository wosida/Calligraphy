import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('work/picture1.png',0)
plt.imshow(img,cmap='gray')
plt.show()