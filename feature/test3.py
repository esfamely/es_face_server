import numpy as np
import cv2
import h5py

img1 = cv2.imread("D:/s5/lena/lena.png")
img2 = cv2.imread("D:/s5/lena/103752.jpg")
img2 = cv2.resize(img2, (512, 512))

d1 = np.reshape(img1, (-1,))
d2 = np.reshape(img2, (-1,))

db = h5py.File("D:/s5/lena/005.hdf5", "w")
data = db.create_dataset("img", d1.shape, dtype="float")
data[0 : len(d1)] = d1
db.close()

book = h5py.File("D:/s5/lena/005.hdf5", "r")
b1 = np.copy(book["img"])
b1_ = np.reshape(b1, (512, 512, 3)).astype(np.uint8)
book.close()

cv2.imshow("es", b1_)
cv2.waitKey(0)

db_ = h5py.File("D:/s5/lena/005.hdf5", "w")
data_ = db_.create_dataset("img", (b1.shape[0] + d2.shape[0],), dtype="float")
data_[0 : b1.shape[0]] = b1
data_[b1.shape[0] :] = d2
db_.close()

look = h5py.File("D:/s5/lena/005.hdf5", "r")
l1 = look["img"][0 : b1.shape[0]]
l2 = look["img"][b1.shape[0] :]
l1_ = np.reshape(l1, (512, 512, 3)).astype(np.uint8)
l2_ = np.reshape(l2, (512, 512, 3)).astype(np.uint8)

cv2.imshow("es", l2_)
cv2.waitKey(0)
cv2.imshow("es", l1_)
cv2.waitKey(0)
