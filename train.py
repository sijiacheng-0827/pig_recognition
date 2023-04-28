# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_wine
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# import os
# from PIL import Image
# from torch.utils.data import Dataset
import csv

# def readimg(filepath, labelpath, idx):
#     imgs = os.listdir(filepath)
#     imgname = imgs[idx]
#     img_item_path = os.path.join(filepath, imgname)
#     img = Image.open(img_item_path)
#     labels = os.listdir(labelpath)
#     labelname = labels[idx]
#     label_item_path = os.path.join(labelpath, labelname)
#     label = Image.open(label_item_path)
#     return img, label
#
#
#
#
# filepath = r'C:\Users\PC\Desktop\pigg\captrue'
# csv_path = r'D:\Zhenong\data\all.csv'
# labelpath = r'C:\Users\PC\Desktop\pigg\captrue_label'


from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import model_selection
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
#  定义字典，便于来解析样本数据集txt
# def Iris_label(s):
#     it = {b'Vegetation':0, b'Non-Vegetation':1}
#     return it[s]

# path = r'D:\Zhenong\data\all.csv'
path = r'D:\Zhenong\data\train.csv'

SavePath = r"D:\model.pickle"


#  1.读取数据集
with open(path, encoding='utf-8') as f:
   data = np.loadtxt(f, dtype=float, delimiter=',')
#       data = csv.reader(fp)
#       list = []
#       for row in data:
#           list.append(row)
#       print(list)
#     # print(data)


#  2.划分数据与标签
x, y = np.split(data, indices_or_sections=(3,), axis=1)  # x为数据，y为标签
# print(x, y)
x = x[:, 0:3]
#
train_data, test_data, train_label, test_label = model_selection.train_test_split(x, y, random_state=1, train_size=0.9, test_size=0.1)
# train_data, test_data, train_label, test_label = model_selection.train_test_split(x, y, random_state=1, train_size=0.5, test_size=0.5)

#  3.用100个树来创建随机森林模型，训练随机森林
classifier = RandomForestClassifier(n_estimators=100,
                               bootstrap=True,
                               max_features='sqrt')

classifier.fit(train_data, train_label.ravel())#ravel函数拉伸到一维
#
#
#  4.计算随机森林的准确率
print("训练集：",classifier.score(train_data,train_label))
print("测试集：",classifier.score(test_data,test_label))
#
#  5.保存模型
#以二进制的方式打开文件：
file = open(SavePath, "wb")
#将模型写入文件：
pickle.dump(classifier, file)
#最后关闭文件：
file.close()