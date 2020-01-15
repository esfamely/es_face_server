import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from feature.facenet_feature_extractor import FacenetFeatureExtractor
from feature.nearest_neighbors_classifier import NearestNeighborsClassifier
import h5py
from main.setup import Setup
from pathlib import Path

dir = Setup.s4_feature_dir
p = 0.1

datas_train = []
labels_train = []
datas_test = []
labels_test = []
target_names = []
for i in range(1, 6):
    hdf5_file_path = "{}/facenet/{:03d}.hdf5".format(Setup.s4_feature_dir, i)
    #hdf5_file_path = "{}/resnet50/{:03d}.hdf5".format(Setup.s4_feature_dir, i)
    d = h5py.File(hdf5_file_path, "r")
    datas = d["imgs"]
    labels = d["labels"]
    index = int(len(datas) * p)
    datas_train.extend(datas[0 : index])
    labels_train.extend(labels[0: index])
    datas_test.extend(datas[index :])
    labels_test.extend(labels[index:])
    target_names.append("{:03d}".format(i))
print("train size: {}, test size: {}".format(len(datas_train), len(datas_test)))

print("[INFO] evaluating classifier ...")
model = KNeighborsClassifier(1, n_jobs=-1)
#model = LogisticRegression()
model.fit(datas_train, labels_train)
preds = model.predict(datas_test)
print(classification_report(labels_test, preds, target_names=target_names))

# compute the raw accuracy with extra precision
acc = accuracy_score(labels_test, preds)
print("[INFO] score: {:.2f}%\n".format(acc * 100))

model_new = NearestNeighborsClassifier(1, distance_threshold=1.05)
model_new.fit2(datas_train, labels_train)
preds_new = model_new.predict2(datas_test)
print(preds_new)
acc_ = accuracy_score(labels_test, preds_new)
print("[INFO] new score: {:.2f}%\n".format(acc_ * 100))

'''feature_extractor = FacenetFeatureExtractor(Setup.s4_facenet_model_path)
lfw_dir = "D:/s5/dataset/lfw_160"
lfw_list = list(Path(lfw_dir).glob("**/*.png"))[0:500]
lfws = [cv2.imread(str(lfw)) for lfw in lfw_list]
lfws_f = feature_extractor.extract_features(lfws, batch_size=16)
preds_lfw, ps_dist = model_new.predict2(lfws_f, return_dist=True)
print(preds_lfw)
for i, p_lfw in enumerate(preds_lfw):
    if p_lfw != -1:
        print("{}: {}".format(lfw_list[i], ps_dist[i]))
        cv2.imshow("lfw", lfws[i])
        cv2.waitKey(0)'''
