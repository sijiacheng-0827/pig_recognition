import os
import numpy as np
from sklearn.cluster import KMeans
import cv2 as cv
from imutils import build_montages
import matplotlib.image as imgplt

image_path = []
all_images = []
images = os.listdir('./data/crop_test_images')

# 将图像裁剪成统一大小，否则下面的代码会报错；这里已经裁剪好了
# for image in images:
#     img = cv.imread('./data/test_images/' + image)
#     img = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
#     cv.imwrite('./data/crop_test_images/' + image, img)

for image_name in images:
    image_path.append('./data/crop_test_images/' + image_name)
for path in image_path:
    image = imgplt.imread(path)
    image = image.reshape(-1, )
    all_images.append(image)

clt = KMeans(n_clusters=2)
clt.fit(all_images)
labelIDs = np.unique(clt.labels_)

for labelID in labelIDs:
    idxs = np.where(clt.labels_ == labelID)[0]
    idxs = np.random.choice(idxs, size=min(25, len(idxs)), replace=False)
    show_box = []
    for i in idxs:
        image = cv.imread(image_path[i])
        image = cv.resize(image, (96, 96))
        show_box.append(image)
    montage = build_montages(show_box, (96, 96), (5, 5))[0]

    title = "Type {}".format(labelID)
    cv.imshow(title, montage)
    cv.waitKey(0)