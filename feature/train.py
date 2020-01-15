import os

import h5py
import numpy as np
from db.table_dict import TableDict

from feature.facenet_feature_extractor import FacenetFeatureExtractor
from feature.hdf5_writer import Hdf5Writer
from main.setup import Setup
from utils.utils_cv import list_imgs_files
from utils.utils_sys import now_dt_str, id_generator, info


class Train:
    """
    模型训练

    es 2018-11-12
    """

    def train(self, uid_to_un, uid_to_label, label_to_uid):
        # Facenet特征提取
        facenet_feature_extractor = FacenetFeatureExtractor(Setup.s4_facenet_model_path)
        self.feature_extract(facenet_feature_extractor, Setup.s4_facenet_batch_size, uid_to_label)

    @staticmethod
    def load_imgs(uid):
        """
        加载某人员的图像
        """
        dir = os.path.join(Setup.s3_face_dir, uid)
        return list_imgs_files(dir)

    def filter_imgs(self, imgs, fs, uid):
        """
        已提取的图像不再重复提取
        """
        imgs_filtered, fs_filtered = [], []

        # 读取数据库提取记录
        table_dict = TableDict()
        sql = "select * from face_train_feature where isdeleted = '0' and uid = %(uid)s"
        fe_list = table_dict.list(sql, {"uid": uid})

        # 过滤已提取的
        for i, f in enumerate(fs):
            is_in = 0
            for fe in fe_list:
                if fe["iid"] == f.split(".")[0]:
                    is_in = 1
                    break
            if is_in == 0:
                imgs_filtered.append(imgs[i])
                fs_filtered.append(f)

        return imgs_filtered, fs_filtered

    def feature_extract(self, feature_extractor, batch_size, uid_to_label):
        """
        特征提取
        """
        table_dict = TableDict()
        tid = id_generator()
        name = "facenet"

        # 记录本次训练情况
        train_dict = {"id": tid, "tt": "1", "tp": 0.0, "sdt": now_dt_str(), "isdeleted": "0"}
        table_dict.save("face_train", train_dict)

        for i, uid in enumerate(uid_to_label):
            imgs, fs = Train.load_imgs(uid)
            imgs, fs = self.filter_imgs(imgs, fs, uid)
            info("uid: {}, len: {}, feature extract ...".format(uid, len(imgs)))
            if len(imgs) == 0:
                info("uid: {}, len: {}, feature extract ok".format(uid, len(imgs)))
                continue

            features = feature_extractor.extract_features(imgs, batch_size=batch_size)
            labels = (np.ones(len(features)) * int(uid_to_label[uid])).astype(np.int32)

            # 特征文件存放路径
            dir = os.path.join(Setup.s4_feature_dir, name)
            if os.path.exists(dir) == False:
                os.makedirs(dir)
            hdf5_file_path = os.path.join(dir, "{}.hdf5".format(uid))

            # 类内部序号
            sn = 0

            # 特征文件若已存在，则读取里面数据，与今次新增的数据集合，一起写入文件
            if os.path.exists(hdf5_file_path):
                db_exist = h5py.File(hdf5_file_path, "r")
                features_exist = np.copy(db_exist["imgs"])
                labels_exist = np.copy(db_exist["labels"])
                db_exist.close()
                sn += features_exist.shape[0]
                info("uid: {}, feature exist {}, now add ...".format(uid, features_exist.shape[0]))

                features_new, labels_new = [], []
                features_new.extend(features_exist)
                features_new.extend(features)
                labels_new.extend(labels_exist)
                labels_new.extend(labels)
                hdf5_writer = Hdf5Writer(np.shape(features_new), hdf5_file_path, dataKey="imgs")
                hdf5_writer.add(features_new, labels_new)
                hdf5_writer.close()
            else:
                hdf5_writer = Hdf5Writer(np.shape(features), hdf5_file_path, dataKey="imgs")
                hdf5_writer.add(features, labels)
                hdf5_writer.close()

            # 保存提取记录到数据库
            to_db_list = []
            for f in fs:
                to_db_list.append({"id": id_generator(), "tid": tid, "uid": uid,
                                   "label": str(uid_to_label[uid]), "iid": f.split(".")[0], "sn": sn,
                                   "dt": now_dt_str(), "isdeleted": "0"})
                sn += 1
            table_dict.batch_save("face_train_feature", to_db_list)

            # 更新训练进度
            train_dict["tp"] = (i + 1) / len(uid_to_label)
            table_dict.update("face_train", train_dict)

            info("uid: {}, len: {}, feature extract ok".format(uid, len(imgs)))

        # 更新训练完成时间
        train_dict["tp"] = 1.0
        train_dict["edt"] = now_dt_str()
        table_dict.update("face_train", train_dict)

'''train = Train()
train.train()'''
