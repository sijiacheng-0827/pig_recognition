import csv

import torch
from PIL import Image
import cv2
import numpy as np
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F
import pandas as pd
### 裁剪出的小猪猪图片，将图片的每个像素点信息读出

# # # 打开要处理的图像
# img = Image.open(r'C:\Users\PC\Desktop\pigonly\1-1.png')
# # pil.Image.open()是rgb格式
# # 如果是cv2.imread() 就是bgr格式
# img_array = np.array(img)  # 把图像转成数组格式img = np.asarray(image)  (H, W, C)=(102, 76, 4)
# # img_array = np.transpose(img_array, (2, 0, 1))  # (C, H, W)=(4, 102, 76)
# # shape = img_array.shape


#### 读取裁剪出的小猪猪的均值和方差
img = cv2.imread(r'C:\Users\PC\Desktop\pigonly\3-7.png', 1)                 # 1是以彩色图方式去读
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image = F.to_tensor(img)
means = [0, 0, 0]
std = [0, 0, 0]
for i in range(3):
    means[i] += img[i, :, :].mean()
    std[i] += img[i, :, :].std()
# print(means)
# print(std)
## opencv储存图片的格式和torch的储存方式不一样，opencv储存图片格式是（H,W,C），
# 而torch储存的格式是（C,H,W）

row = []  # rgb值
r1 = []  # r值
r2 = []  # g值
r3 = []  # b值
m = []
s = []

# headers = ['color', 'mean', 'std', 'ispig']

for i in range(img.shape[0]):  # 读取前三个通道的像素值
    for j in range(img.shape[1]):
        lst = img[i][j].tolist()
        if sum(lst) != 0:
            row.append(lst)
            r1.append(lst[0])
            r2.append(lst[1])
            r3.append(lst[2])
# 计算均值
r1_mean = np.array(r1).mean()
r2_mean = np.array(r2).mean()
r3_mean = np.array(r3).mean()
# r_mean = [r1_mean, r2_mean, r3_mean]

# 计算标准差
r1_std = np.array(r1).std()
r2_std = np.array(r2).std()
r3_std = np.array(r3).std()
# r_std = [r1_std, r2_std, r3_std]


# for i in row:  # mean和std的数量要跟row的一致才能写入csv文件
# m.append(r_mean)
# s.append(r_std)

# dataframe 转csv方法
# dataframe = pd.DataFrame({'color': row, 'mean': m, 'std': s, 'ispig': None})
# dataframe = pd.DataFrame({'mean': m, 'std': s, 'ispig': 1})
# dataframe.to_csv('pig7.csv', mode='a')

# with open(r'D:\Zhenong\data\all.csv', 'a', newline='') as f:
#     csv_writer = csv.writer(f)
#     csv_writer.writerow([r_mean, r_std, 1])

with open(r'D:\Zhenong\data\all.csv', 'a', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow([r1_mean, r2_mean, r3_mean, r1_std, r2_std, r3_std, 0])
