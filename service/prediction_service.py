import datetime as dt
import os

import cv2
import numpy as np
from db.table_dict import TableDict

from main.setup import Setup
from utils.utils_cv import distance_l2, add_border
from utils.utils_sys import now_dt_str, id_generator


class PredictionService:
    """
    分类识别
    """

    def __init__(self, collection_service, training_service):
        self.collection_service = collection_service
        self.training_service = training_service

    def predict(self, cid, pid, fs):
        """
        人脸识别
        :param cid: 采集id
        :param pid: 终端id
        :param fs: 人脸图文件
        :return: {'uid': 识别结果（人员id）, 'cc': 成功添加的样本数}
        """
        # 加载训练信息
        prediction_feature = self.training_service.get_prediction_feature()
        uid_to_un = self.training_service.get_uid_to_un()
        uid_to_label = self.training_service.get_uid_to_label()
        label_to_uid = self.training_service.get_label_to_uid()

        # 计算识别耗时
        start_time = dt.datetime.now()

        # 临时保存目录
        tmp_dir = Setup.s3_face_dir + "_tmp/"
        if os.path.exists(tmp_dir) == False:
            os.makedirs(tmp_dir)

        # 将文件保存到本地
        imgs, filenames = [], []
        for f in fs:
            filename = id_generator() + ".jpg"
            file_path = os.path.join(tmp_dir, filename)
            f.save(file_path)
            imgs.append(cv2.imread(file_path))
            filenames.append(filename)

        # 数据清洗
        imgs, filenames = self.collection_service.get_data_processor().data_wash(imgs, filenames)
        # 数据预处理
        imgs = self.collection_service.get_data_processor().data_preprocess(imgs)

        # 进行识别
        preds, ps_dist, ps_sim, ps_sn = [], [], [], []
        if Setup.s4_use_feature_extract == 1:
            preds, ps_dist, ps_sim, ps_sn = prediction_feature.prediction(imgs)

        # 相似度最高的作为识别结果
        index = int(np.argmax(ps_sim))
        pred = preds[index]
        sim = float(np.max(ps_sim))
        filename = filenames[index]
        uid = "0" if pred == -1 else label_to_uid[str(pred)]

        # 在数据集里找到当前识别对象的最新图像
        miid = ""
        if uid != "0":
            u_newest = self.collection_service.get_newest_by_uid(uid)
            miid = u_newest["iid"]

        # 识别耗时
        times = (dt.datetime.now() - start_time)
        tc = float(times.seconds * np.power(10, 3) + times.microseconds / np.power(10, 3))

        # 记录识别结果
        table_dict = TableDict()
        iniid = filename.split(".")[0]
        dict = {"id": id_generator(), "cid": cid, "pid": pid, "puid": uid, "tuid": uid, "sim": sim,
                "iniid": iniid, "miid": miid, "tc": tc, "dt": now_dt_str(), "isdeleted": "0"}
        table_dict.save("face_prediction", dict)

        # 识别结果对应的图像加入数据集
        imgs_p, filenames_p = [], []
        for i, p in enumerate(preds):
            if p == pred:
                imgs_p.append(imgs[i])
                filenames_p.append(filenames[i])
        json_r = self.collection_service.collect(cid, pid, uid, None,
                                                 ct="2", imgs=imgs_p, filenames=filenames_p)
        json_r["uid"] = uid

        return json_r

    def get_face_by_rectangle(self, img, rectangle):
        """
        指定人脸框位置得到人脸区域图
        """
        if rectangle is None:
            return img
        else:
            rs = rectangle.split(",")
            if len(rs) >= 4:
                img_width, img_height = img.shape[1], img.shape[0]
                top, left, width, height = int(rs[0]), int(rs[1]), int(rs[2]), int(rs[3])
                if top >= img_height:
                    return img
                elif (top + height) > img_height:
                    height = img_height - top
                if left >= img_width:
                    return img
                elif (left + width) > img_width:
                    width = img_width - left
                return img[top : top + height, left : left + width, :]
        return img

    def compare(self, image_url1, image_file1, image_base64_1, face_rectangle1,
                image_url2, image_file2, image_base64_2, face_rectangle2):
        """
        1 vs 1 人脸比对
        :param image_url1: 图片的 URL
        :param image_file1: 一个图片，二进制文件
        :param image_base64_1: base64 编码的二进制图片数据
        :param face_rectangle1: 人脸矩形框的位置
        """
        # 返回信息
        r = {}
        request_id = id_generator()
        image_id1 = id_generator()
        image_id2 = id_generator()
        r["request_id"] = request_id

        # 计算耗时
        start_time = dt.datetime.now()

        # 临时保存目录
        tmp_dir = Setup.s3_face_dir + "_api/"
        if os.path.exists(tmp_dir) == False:
            os.makedirs(tmp_dir)

        # 将文件保存到本地
        file_path1 = self.collection_service.get_request_img(tmp_dir, image_id1,
                                                             image_url1, image_file1, image_base64_1)
        file_path2 = self.collection_service.get_request_img(tmp_dir, image_id2,
                                                             image_url2, image_file2, image_base64_2)

        # 读取图像
        img1 = None
        img2 = None
        if file_path1 is None:
            r["error_message"] = "没有传入第一张图片"
            return r
        elif file_path2 is None:
            r["error_message"] = "没有传入第二张图片"
            return r
        else:
            img1 = cv2.imread(file_path1)
            img2 = cv2.imread(file_path2)
            r["image_id1"] = image_id1
            r["image_id2"] = image_id2

        # 当传入图片进行人脸检测时，是否指定人脸框位置进行检测
        img1 = self.get_face_by_rectangle(img1, face_rectangle1)
        img2 = self.get_face_by_rectangle(img2, face_rectangle2)

        # 如果人脸刚好占满整个图像区域，目前的人脸检测方法可能会漏检，加个边框会大大减少漏检的发生
        img1 = add_border(img1)
        img2 = add_border(img2)

        # 数据清洗
        imgs = [img1, img2]
        imgs, _ = self.collection_service.get_data_processor().data_wash(imgs)
        if len(imgs) < 2:
            r["error_message"] = "传入的两张图片中，找不到足够两个的人脸来进行分析"
            return r
        # 数据预处理
        imgs = self.collection_service.get_data_processor().data_preprocess(imgs)

        # 特征提取
        prediction_feature = self.training_service.get_prediction_feature()
        feature_extractor = prediction_feature.get_feature_extractor()
        features = feature_extractor.extract_features(imgs)

        # 样本距离与相似度
        distance = distance_l2(features[0], features[1])
        confidence = prediction_feature.cal_sim([distance])[0]
        thresholds = Setup.s4_distance_threshold
        r["distance"] = str(distance)
        r["confidence"] = str(confidence)
        r["thresholds"] = str(thresholds)

        # 耗时
        times = (dt.datetime.now() - start_time)
        time_used = times.seconds * np.power(10, 3) + times.microseconds / np.power(10, 3)
        r["time_used"] = time_used

        return r

    def search(self, image_url, image_file, image_base64, face_rectangle):
        """
        1 vs n 人脸检索
        :param image_url: 图片的 URL
        :param image_file: 一个图片，二进制文件
        :param image_base64: base64 编码的二进制图片数据
        :param face_rectangle: 人脸矩形框的位置
        """
        # 加载训练信息
        prediction_feature = self.training_service.get_prediction_feature()
        uid_to_un = self.training_service.get_uid_to_un()
        uid_to_label = self.training_service.get_uid_to_label()
        label_to_uid = self.training_service.get_label_to_uid()

        # 返回信息
        r = {}
        request_id = id_generator()
        image_id = id_generator()
        r["request_id"] = request_id

        # 计算耗时
        start_time = dt.datetime.now()

        # 临时保存目录
        tmp_dir = Setup.s3_face_dir + "_api/"
        if os.path.exists(tmp_dir) == False:
            os.makedirs(tmp_dir)

        # 将文件保存到本地
        file_path = self.collection_service.get_request_img(tmp_dir, image_id,
                                                             image_url, image_file, image_base64)

        # 读取图像
        img = None
        if file_path is None:
            r["error_message"] = "没有传入图片"
            return r
        else:
            img = cv2.imread(file_path)
            r["image_id"] = image_id

        # 当传入图片进行人脸检测时，是否指定人脸框位置进行检测
        img = self.get_face_by_rectangle(img, face_rectangle)

        # 如果人脸刚好占满整个图像区域，目前的人脸检测方法可能会漏检，加个边框会大大减少漏检的发生
        img = add_border(img)

        # 数据清洗
        imgs = [img]
        imgs, _ = self.collection_service.get_data_processor().data_wash(imgs)
        if len(imgs) < 1:
            r["error_message"] = "传入的图片中，找不到人脸来进行分析"
            return r
        # 数据预处理
        imgs = self.collection_service.get_data_processor().data_preprocess(imgs)

        # 进行识别
        preds, ps_dist, ps_sim, ps_sn = prediction_feature.prediction(imgs)
        if len(preds) == 0:
            r["error_message"] = "模型尚未训练，暂无法进行识别"
            return r
        uid = "0" if preds[0] == -1 else label_to_uid[preds[0]]
        un = "陌生人" if uid == "0" else uid_to_un[uid]
        r["uid"] = uid
        r["un"] = un

        # 样本距离与相似度
        distance = ps_dist[0]
        confidence = ps_sim[0]
        thresholds = Setup.s4_distance_threshold
        r["distance"] = str(distance)
        r["confidence"] = str(confidence)
        r["thresholds"] = str(thresholds)

        # 耗时
        times = (dt.datetime.now() - start_time)
        time_used = times.seconds * np.power(10, 3) + times.microseconds / np.power(10, 3)
        r["time_used"] = time_used

        return r

    def load_recent_prediction(self, hm):
        """
        加载最近的识别结果
        """
        table_dict = TableDict()

        sql = "select fp.*, fcu.un as pun from face_prediction fp, face_collection_users fcu"
        sql += " where fp.isdeleted = '0' and fcu.isdeleted = '0' and fp.puid = fcu.uid"
        sql += " order by fp.dt desc limit 0, %(hm)s"

        p_list = table_dict.list(sql, {"hm": hm})
        for p in p_list:
            p["dt"] = str(p["dt"])
            # 待识别与匹配的图像地址
            img_path1 = str(p["puid"]) + "/" + p["iniid"] + ".jpg"
            p["in_img_url"] = Setup.s3_face_dir[6:] + "/" + img_path1
            img_path2 = str(p["puid"]) + "/" + p["miid"] + ".jpg"
            p["m_img_url"] = Setup.s3_face_dir[6:] + "/" + img_path2

        return p_list
