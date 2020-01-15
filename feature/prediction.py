from feature.nearest_neighbors_classifier import NearestNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import h5py
from main.setup import Setup
from utils.utils_sys import info
from feature.facenet_feature_extractor import FacenetFeatureExtractor
import os
from pathlib import Path


class Prediction:
    """
    model分类预测
    es 2018-11-13
    """

    def __init__(self, p=1.0):
        """
        :param p: 每个分类取多少比例的样本进行训练
        """
        self.model = None

        self.load_model("facenet", p)

        # 特征提取器
        self.feature_extractor = FacenetFeatureExtractor(Setup.s4_facenet_model_path)

    def get_feature_extractor(self):
        return self.feature_extractor

    def load_model(self, name, p):
        """
        加载model
        """
        self.datas_train = []
        self.labels_train = []
        self.datas_test = []
        self.labels_test = []
        dir = os.path.join(Setup.s4_feature_dir, name)
        files = list(Path(dir).glob("**/*.hdf5"))
        for file in files:
            db = h5py.File(str(file), "r")
            datas = db["imgs"]
            labels = db["labels"]
            index = int(len(datas) * p)
            self.datas_train.extend(datas[0: index])
            self.labels_train.extend(labels[0: index])
            self.datas_test.extend(datas[index:])
            self.labels_test.extend(labels[index:])

        if len(self.datas_train) > 0:
            if name == "facenet":
                self.model = NearestNeighborsClassifier(1,
                                                        distance_threshold=Setup.s4_distance_threshold)
                self.model.fit(self.datas_train, self.labels_train)

        self.datas_train.clear()
        # labels_train需要保留，分类时有用
        #self.labels_train.clear()

    def check_model(self):
        """
        确认model是否成功加载
        """
        if self.model is None:
            info("training first")
            return False
        else:
            if len(self.datas_test) == 0:
                info("no test data")
                return False

            info("evaluating classifier ...")
            preds = self.model.predict(self.datas_test)
            acc = accuracy_score(self.labels_test, preds)
            info("score: {:.2f}%\n".format(acc * 100))

            self.datas_test.clear()
            self.labels_test.clear()

            return acc > 0.5

    def prediction(self, samples, batch_size=Setup.s4_facenet_batch_size):
        """
        分类预测
        """
        if self.model is None:
            info("training first")
            return [], [], [], []

        features = self.feature_extractor.extract_features(samples, batch_size=batch_size)
        preds, ps_dist, ps_sn = self.model.predict(features, return_dist=True)
        ps_sim = self.cal_sim(ps_dist)
        return preds, ps_dist, ps_sim, ps_sn

    def cal_sim(self, ps_dist):
        """
        计算相似度
        :param ps_dist: 样本距离
        """
        ps_sim = []
        for dist in ps_dist:
            sim = 1 - dist / (2*Setup.s4_distance_threshold)
            ps_sim.append(sim)
        return ps_sim


'''prediction = Prediction(p=0.007)
print(prediction.check_model())
import cv2
samples = []
dir = "D:/s5/dataset/easyface"
ds = os.listdir(dir)
for d in ds:
    files = list(Path(dir + "/" + str(d)).glob("**/*.jpg"))
    for file in files[-10:]:
        samples.append(cv2.imread(str(file)))
preds = prediction.prediction(samples)
print(preds)'''
