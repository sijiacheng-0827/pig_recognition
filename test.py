import numpy as np
import pickle
import cv2
import torchvision.transforms.functional as F
from PIL import Image


RFpath = r"D:\model.pickle"
# Landset_Path = r"vis.jpg"
# SavePath = r"D:\save.jpg"
################################################调用保存好的模型
#以读二进制的方式打开文件
file = open(RFpath, "rb")
#把模型从文件中读取出来
rf_model = pickle.load(file)
#关闭文件
file.close()
################################################用读入的模型进行预测
img = cv2.imread(r'pig.jpg', 1)                # 1是以彩色图方式去读
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image = img.transpose(2,0,1)
data = np.zeros((image.shape[0],image.shape[1]*image.shape[2]))
for i in range(image.shape[0]):
    data[i] = image[i].flatten()
data = data.swapaxes(0,1)
pred = rf_model.predict(data)
pred = pred.reshape(image.shape[1],image.shape[2])*255
pred = pred.astype(np.uint8)
im = Image.fromarray(pred)
im.save("test.jpg")
im.show()