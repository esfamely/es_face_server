from db.table_dict import TableDict

from feature.prediction import Prediction as PredictionFeature
from feature.train import Train as TrainFeature
from main.setup import Setup


class TrainingService:
    """
    模型训练
    """

    def __init__(self):
        self.prediction_feature = PredictionFeature()
        self.uid_to_un, self.uid_to_label, self.label_to_uid = self.map_uid_un_label()

    def get_prediction_feature(self):
        return self.prediction_feature

    def get_uid_to_un(self):
        return self.uid_to_un

    def get_uid_to_label(self):
        return self.uid_to_label

    def get_label_to_uid(self):
        return self.label_to_uid

    def map_uid_un_label(self):
        """
        建立人员信息映射
        :return: 映射表
        """
        uid_to_un, uid_to_label, label_to_uid = {}, {}, {}

        # 读取采集人员表
        table_dict = TableDict()
        list_users = table_dict.list("select * from face_collection_users where isdeleted = '0'")
        for user in list_users:
            uid_to_un[user["uid"]] = user["un"]
            uid_to_label[user["uid"]] = user["label"]
            label_to_uid[user["label"]] = user["uid"]

        return uid_to_un, uid_to_label, label_to_uid

    def train(self):
        """
        模型训练
        """
        # 特征提取训练
        if Setup.s4_use_feature_extract == 1:
            train_feature = TrainFeature()
            self.uid_to_un, self.uid_to_label, self.label_to_uid = self.map_uid_un_label()
            train_feature.train(self.uid_to_un, self.uid_to_label, self.label_to_uid)

            # 重新加载特征提取分类器
            self.prediction_feature = PredictionFeature()

    def get_train_info(self):
        """
        得到最近一次训练的情况
        """
        table_dict = TableDict()
        sql = "select * from face_train"
        sql += " where id = (select max(id) from face_train where isdeleted = '0')"
        result_dict_list = table_dict.list(sql)
        if len(result_dict_list) == 0:
            return {}
        for result_dict in result_dict_list:
            result_dict["sdt"] = "" if result_dict["sdt"] is None else str(result_dict["sdt"])
            result_dict["edt"] = "" if result_dict["edt"] is None else str(result_dict["edt"])
        return result_dict_list[0]
