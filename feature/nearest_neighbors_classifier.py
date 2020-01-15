import numpy as np
from sklearn.neighbors import NearestNeighbors


class NearestNeighborsClassifier(NearestNeighbors):
    """
    最近邻分类器

    sklearn自有的最近邻分类器不支持距离阀值，在此封装一个

    es 2018-11-14
    """

    def __init__(self, k, distance_threshold=1.0):
        NearestNeighbors.__init__(self, k)

        self.distance_threshold = distance_threshold

    def fit(self, datas_train, labels_train=None):
        super().fit(datas_train)

        self.labels_train = labels_train

    def fit2(self, datas_train, labels_train):
        self.datas_train = datas_train
        self.labels_train = labels_train

    def predict(self, datas, distance_threshold=None, unknown_label=-1, return_dist=False):
        """
        分类预测，大于距离阀值的样本被统一标记为未知（某指定整数）
        :param distance_threshold: 距离阀值
        """
        if distance_threshold is None:
            distance_threshold = self.distance_threshold

        ps = self.kneighbors(datas)
        ps = np.reshape(ps, (2, -1))
        ps_dist = ps[0]
        ps_ind = ps[1]
        preds = []
        ps_sn = []
        for i, p in enumerate(ps_ind):
            sn = -1

            if ps_dist[i] < distance_threshold:
                sn = int(p)
                preds.append(self.labels_train[sn])
            else:
                preds.append(unknown_label)

            ps_sn.append(sn)

        if return_dist:
            return preds, ps_dist, ps_sn
        else:
            return preds

    def predict2(self, datas, distance_threshold=None, unknown_label=-1, return_dist=False):
        """
        类似predict，但不是直接跟所有样本比较，而跟每类的所有样本距离取均值再进行比较
        :param distance_threshold: 距离阀值
        """
        if distance_threshold is None:
            distance_threshold = self.distance_threshold

        preds = []
        ps_dist = []

        for data in datas:
            dist_list_dict = {}
            for i, label_train in enumerate(self.labels_train):
                dist = np.linalg.norm(np.subtract(data, self.datas_train[i]))

                dist_list = dist_list_dict.get(label_train)
                if dist_list is None:
                    dist_list_dict[label_train] = [dist]
                else:
                    dist_list.append(dist)

            label_list = [label for label in dist_list_dict]
            mean_list = []
            for label in label_list:
                dist_list = dist_list_dict.get(label)
                dist_mean = np.mean(dist_list)
                mean_list.append(dist_mean)

            mean_min = np.min(mean_list)
            if mean_min < distance_threshold:
                index_min = np.argmin(mean_list)
                pred = label_list[index_min]
                preds.append(pred)
            else:
                preds.append(unknown_label)

            ps_dist.append(mean_min)

        if return_dist:
            return preds, ps_dist
        else:
            return preds
