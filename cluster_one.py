import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


# 目的：读取图片，利用【l* a* b*】进行聚类，将聚类中心保存到npy文件，在画出的色度图上画出这些聚类中心


# 读取指定文件夹下所有文件的名字
def readname(filepath):
    name = os.listdir(filepath)
    return name


# 利用[l* a* b*]实现聚类
def seg_kmeans_color(img):
    # img = cv2.imread('D:\\MATLABfile\\2020-9-24\\resize.png', cv2.IMREAD_COLOR) # cv2读入三通道图像的顺序是BGR
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # BGR转到LAB空间
    # 变换图像通道bgr->rgb
    # b, g, r = cv2.split(img)
    # img = cv2.merge([r, g, b])
    # 3个通道展平 二维矩阵展成一维向量
    img_flat = img.reshape((img.shape[0] * img.shape[1], 3))
    img_flat = np.float32(img_flat)
    # 迭代参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # 这里k=4
    compactness, labels, centers = cv2.kmeans(img_flat, 8, None, criteria, 10, flags)
    # 显示结果
    img_output = labels.reshape((img.shape[0], img.shape[1]))
    plt.subplot(121), plt.imshow(img), plt.title('input')
    plt.subplot(122), plt.imshow(img_output), plt.title('kmeans')
    plt.show()
    print(centers)

    return centers


#
# cv2.cvtColor(img, cv2.COLOR_LAB2RGB)


if __name__ == '__main__':

    img_name = './data/vis.jpg'
    img = cv2.imread(img_name)  # cv2读入顺序是bgr
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    # img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # RGB颜色空间转换到LAB颜色空间
    centers = seg_kmeans_color(img)

'''
注意点：cv2.imread(path)  # 读入三通道图的顺序是bgr

from PIL import Image

Image.open(path)  # 读入三通道图像的顺序是rgb

a = [];  # 建立空列表
np.array(a);  # 列表转换为数组，注意空列表无法这样转换为数组
a.tolist();  # 转换为list
'''

