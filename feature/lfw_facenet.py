import numpy as np
from feature.lfw_read_images import LfwReadImages
import feature.calculate_threshold as calculate_threshold
from feature.mini_facenet_feature_extractor import MiniFacenetFeatureExtractor
from feature.facenet_feature_extractor import FacenetFeatureExtractor
from feature.hdf5_writer import Hdf5Writer
import h5py
import os

root_dir = "D:/s5/dataset/"
lfw_path = root_dir + "lfw_160/"
pairs_train_path = root_dir + "lfw_txt/pairsDevTrain.txt"
pairs_vat_path = root_dir + "lfw_txt/pairsDevTest.txt"
pairs_test_path = root_dir + "lfw_txt/pairs.txt"

#model_path = root_dir + "lfw_txt/facenet/160_40_1_77_4.40.ckpt"
#feature_extractor = MiniFacenetFeatureExtractor(model_path)
model_path = "D:/s5/cv_python/facenet/models/20180402-114759/20180402-114759.pb"
feature_extractor = FacenetFeatureExtractor(model_path)

hdf5_dir = "D:/s5/cv_python/es_face/feature/hdf5/"


def extract_features(img_s1, img_s2, name="e_f"):
    img_ts1, img_ts2 = [], []

    hdf5_file_path1 = os.path.join(hdf5_dir, "lfw_{}_1.hdf5".format(name))
    if os.path.exists(hdf5_file_path1):
        db_exist = h5py.File(hdf5_file_path1, "r")
        img_ts1 = np.copy(db_exist["imgs"])
        db_exist.close()
    else:
        img_ts1 = feature_extractor.extract_features(img_s1, batch_size=16)

        hdf5_writer = Hdf5Writer(np.shape(img_ts1), hdf5_file_path1, dataKey="imgs")
        hdf5_writer.add(img_ts1, np.zeros(len(img_s1)))
        hdf5_writer.close()

    hdf5_file_path2 = os.path.join(hdf5_dir, "lfw_{}_2.hdf5".format(name))
    if os.path.exists(hdf5_file_path2):
        db_exist = h5py.File(hdf5_file_path2, "r")
        img_ts2 = np.copy(db_exist["imgs"])
        db_exist.close()
    else:
        img_ts2 = feature_extractor.extract_features(img_s2, batch_size=16)

        hdf5_writer = Hdf5Writer(np.shape(img_ts2), hdf5_file_path2, dataKey="imgs")
        hdf5_writer.add(img_ts2, np.zeros(len(img_s2)))
        hdf5_writer.close()

    return img_ts1, img_ts2


def list_metrics(img_ts1, img_ts2, labels, name="e_f"):
    ms = []
    gd = 0
    gd1 = 0
    gd2 = 0
    for i in range(len(labels)):
        img_t1 = img_ts1[i]
        img_t2 = img_ts2[i]
        d = metrics(img_t1, img_t2)
        ms.append(d)
        gd += d
        if labels[i] == 0:
            gd1 += d
        if labels[i] == 1:
            gd2 += d
        if i % 100 == 0:
            print("{}: {:03d}: d: {:.6f}".format(name, i, gd / 100))
            gd = 0
    print("gd1: d: {:.6f}".format(gd1))
    print("gd2: d: {:.6f}".format(gd2))
    return ms


def metrics(x1, x2):
    d = np.sqrt(np.sum(np.square(np.subtract(x1, x2))))
    return d


img_s1_train, img_s2_train, labels_train = LfwReadImages.read_images(lfw_path, pairs_train_path)
train_size = len(labels_train)
print("train_size: {}".format(train_size))

img_s1_vat, img_s2_vat, labels_vat = LfwReadImages.read_images(lfw_path, pairs_vat_path)
vat_size = len(labels_vat)
print("vat_size: {}".format(vat_size))

img_s1_test, img_s2_test, labels_test = LfwReadImages.read_images(lfw_path, pairs_test_path)
test_size = len(labels_test)
print("test_size: {}".format(test_size))

name_train = "train"
name_vat = "vat"
name_test = "test"

img_ts1_train, img_ts2_train = extract_features(img_s1_train, img_s2_train, name_train)
img_ts1_vat, img_ts2_vat = extract_features(img_s1_vat, img_s2_vat, name_vat)
img_ts1_test, img_ts2_test = extract_features(img_s1_test, img_s2_test, name_test)

ds_train = list_metrics(img_ts1_train, img_ts2_train, labels_train, name_train)
ds_vat = list_metrics(img_ts1_vat, img_ts2_vat, labels_vat, name_vat)
ds_test = list_metrics(img_ts1_test, img_ts2_test, labels_test, name_test)

# show accuracy
mean1, mean2 = calculate_threshold.statistics_from_train(ds_train, labels_train)
print("{} - {}".format(mean1, mean2))
threshold_best_vat, acc_max_vat, range = calculate_threshold.calculate_threshold_from_vat(
    ds_vat, labels_vat, mean1, mean2, num=100)
print("best_vat: {} - {}".format(threshold_best_vat, acc_max_vat))
threshold_best, acc_max = calculate_threshold.show_from_test(
    ds_test, labels_test, threshold_best_vat, range, num=100)
print("best: {} - {}".format(threshold_best, acc_max))
