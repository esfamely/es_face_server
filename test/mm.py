import numpy as np
import cv2
from dataset.mtcnn.mtcnn import Mtcnn as FaceDetector
from feature.facenet_feature_extractor import FacenetFeatureExtractor
from main.setup import Setup

fd = FaceDetector()
ffe = FacenetFeatureExtractor(Setup.s4_facenet_model_path)

img1 = cv2.imread("D:/s5/lena/lena.png")
img2 = cv2.imread("D:/s5/lena/103708.jpg")

tb1, _ = fd.detect_face(img1)
tb2, _ = fd.detect_face(img2)

tb1 = tb1[0]
tb2 = tb2[0]

# 提取人脸图像
if1 = cv2.resize(img1[int(tb1[1]): int(tb1[3]), int(tb1[0]): int(tb1[2]), :], (160, 160))
if2 = cv2.resize(img2[int(tb2[1]): int(tb2[3]), int(tb2[0]): int(tb2[2]), :], (160, 160))

f1 = ffe.extract_feature(if1)
f2 = ffe.extract_feature(if2)
print(np.sqrt(np.sum(np.square(np.subtract(f1, f2)))))
