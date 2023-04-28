import os
from PIL import Image
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
#  把原图和sam图合并
# img = Image.open(r'C:\Users\PC\Desktop\pigg\captrue\K68068135_1_20230419T084141Z.jpg')
# label = Image.open('label.jpg')
# label = label.convert("RGB")
# fin = Image.blend(img,label,0.3)
# fin.show()
# fin.save("./samaddimg.jpg")
# 将随机森林的掩膜与原图结合
image = cv2.imread(r'label.jpg')  # (1440, 2560, 3)
pig = cv2.imread(r'img.jpg')
# sam = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
tes = cv2.imread('test.jpg')  # (1440, 2560, 3)
# gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
masked = cv2.bitwise_and(image, tes)  # (1440, 2560, 3)
# cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
# cv2.imshow('mask', masked)
# cv2.waitKey(0)


# 展示
# cv2.imshow("Mask Applied to Image", masked)
# cv2.waitKey(0)
# cv2.imwrite('samaddforest.jpg', masked)

# for i in range(gray.shape[0]):
#     for j in range(gray.shape[1]):
#         if (gray[i][j] == 0):
#             gray[i][j] = 0
#         else:
#             gray[i][j] = 255
# # data = np.array(gray, dtype='uint8')
# cv2.imwrite('111.jpg', gray)

# 二值化
# img = cv2.imread('./111.jpg')
img_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
# cv2.imshow('image_gray', img_gray)
ret, mask = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY)
# cv2.namedWindow('image_mask', cv2.WINDOW_NORMAL)
# cv2.imshow('image_mask', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# cv2.namedWindow('Binarization', cv2.WINDOW_NORMAL)
# cv2.imshow("Binarization", img_gray)
# cv2.waitKey(0)

# img = cv2.imread('./111.jpg')
# img = np.array(img).shape
# print(img)

# 寻找轮廓
# contours, hierarchy = cv2.findContours('111.jpg',  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# cv2.drawContours(mask,contours,-1,(0,0,255),1)
imgnew = cv2.drawContours(pig, contours, -1, (0, 0, 255), 3)#第三个参数是1，代表了取出contours中index为1的array
cv2.namedWindow('contour', cv2.WINDOW_NORMAL)
cv2.imshow('contour', imgnew)
cv2.waitKey(0)
cv2.destroyAllWindows()