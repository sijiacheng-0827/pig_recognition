import numpy as np
import pickle
import cv2
# import torchvision.transforms.functional as F
# from PIL import Image


# RFpath = r"model.pickle"
# # Landset_Path = r"vis.jpg"
# # SavePath = r"D:\save.jpg"
# ################################################调用保存好的模型
# #以读二进制的方式打开文件
# file = open(RFpath, "rb")
# #把模型从文件中读取出来
# rf_model = pickle.load(file)
# #关闭文件
# file.close()
# ################################################用读入的模型进行预测
# img = cv2.imread(r'pig.jpg', 1)                # 1是以彩色图方式去读
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# h, w, c = img.shape
# img = img.transpose(2,0,1)
# # img = img.reshape(c, h * w)
# data = np.zeros((c, h * w))
# for i in range(c):
#     data[i] = img[i].flatten()
# data = data.swapaxes(0,1)
# pred = rf_model.predict(data)
# t = open('pred.txt', 'w')
# t.write(str(pred))
# t.close()
# pred = (pred.reshape(h, w)).astype(int)
# pred = cv2.imread('mask_pred.jpg', 0)
# mask = cv2.imread('mask.jpg', 0)
# img = cv2.imread('pig.jpg')
# p_arr = np.zeros(256)
# m_p_arr = np.zeros(256)
# # mask_pred = (mask * pred)

# for m_p, m in zip(mask, pred):
    
#     m_p_arr[m_p] += 1
#     p_arr[m] += 1
# for m_p, p in zip(m_p_arr, p_arr):
#     print(m_p, '-----------', p)
# for index in range(len(p_arr)):
#     # if (p * 0.1 / (m_p * 0.1)) < 0.5:
#     if m_p_arr[index] != 0:
#         if (p_arr[index] / m_p_arr[index] < 0.5):
#             print(m_p_arr[index])   
#             m_p_arr[index] = 0
# h, w = pred.shape
# for x in range(h):
#     for y in range(w):
#         a = pred[x][y]
#         if m_p_arr[a] == 0:
#             pred[x][y] == 0 

pred = cv2.imread('pred.jpg', 0)
mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('pig.jpg')
pred_mask = pred >= 100
pred_mask = mask * pred_mask
m_arr = np.zeros(256)
m_p_arr = np.zeros(256)
m_f = mask.flatten()
m_p_f = pred_mask.flatten()
for m, m_p in zip(m_f, m_p_f):
    m_arr[m] += 1
    m_p_arr[m_p] += 1

m_arr_c = m_arr.copy()

for index in range(256):
    if m_arr[index] != 0 and (m_p_arr[index] * 1.0 / m_arr[index] < 0.5):
        m_arr[index] = 0

# a = 0
# for i in range(256):
#     print(m_arr[i], '---------', i)
#     if m_arr[i] != 0: # and m_arr[i] == m_arr_c[i]:
#     # print(m_arr[i], '--------------', m_arr_c[i])
#         a += 1
# print(a)

h, w = pred.shape
for i in range(h * w):
    if m_arr[(m_f[i])] == 0:
        m_f[i] = 0

mask_o = m_f.reshape(h, w)
mask_o = cv2.cvtColor(mask_o, cv2.COLOR_GRAY2BGR)
img = (img * 0.5 + mask_o * 0.9)
# for x in range(h):
#     for y in range(w):
#         if m_arr[(pred[x][y])] == 0:
#             mask_o[x][y] = 0
cv2.imwrite("mask_pred_3.jpg", img)
# cv2.waitKey(0)
