import numpy as np
import cv2
from feature.mini_facenet_feature_extractor import MiniFacenetFeatureExtractor
from feature.facenet_feature_extractor import FacenetFeatureExtractor
from feature.vgg16_feature_extractor import VGG16FeatureExtractor

mini_model_path = "D:/s5/dataset/lfw_txt/facenet/160_40_1_20_0.00.ckpt"
mini_facenet_feature_extractor = MiniFacenetFeatureExtractor(mini_model_path)
model_path = "D:/s5/cv_python/facenet/models/20180402-114759/20180402-114759.pb"
facenet_feature_extractor = FacenetFeatureExtractor(model_path)
vgg16_feature_extractor = VGG16FeatureExtractor()

img = cv2.imread("D:/s5/dataset/lfw_160/Earl_Counter/Earl_Counter_0001.png")
#img = cv2.resize(img, (40, 40))

f_mini_facenet = mini_facenet_feature_extractor.extract_feature(img)
f_facenet = facenet_feature_extractor.extract_feature(img)
f_vgg16 = vgg16_feature_extractor.extract_feature(img)

print(np.mean(f_mini_facenet))
print(np.mean(f_facenet))
print(np.mean(f_vgg16))
