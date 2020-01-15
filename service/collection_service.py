import datetime as dt
import os
from pathlib import Path

import cv2
import numpy as np
from db.table_dict import TableDict

from dataset.data_processor import DataProcessor
from main.setup import Setup
from utils.utils_sys import now_dt_str, info, id_generator
from utils.utils_web import download_file


class CollectionService:
    """
    数据采集
    """

    def __init__(self):
        self.data_processor = DataProcessor()

    def get_data_processor(self):
        return self.data_processor

    def get_request_img(self, tmp_dir, image_id, image_url, image_file, image_base64):
        """
        根据不同图像请求方式，返回最终到达服务器的图像文件路径
        """
        file_path = None
        to_path = os.path.join(tmp_dir, image_id + ".jpg")
        if image_base64 is not None:
            file_path = to_path
            image_base64.save(file_path)
        elif image_file is not None:
            file_path = to_path
            image_file.save(file_path)
        elif image_url is not None:
            file_path = download_file(image_url, to_path)
        return file_path

    def detect(self, image_url, image_file, image_base64, return_landmark, factor_mini):
        """
        人脸检测
        :param image_url: 图片的 URL
        :param image_file: 一个图片，二进制文件
        :param image_base64: base64 编码的二进制图片数据
        :param return_landmark: 是否检测并返回人脸关键点
        :param factor_mini: 缩放因子
        """
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
        file_path = self.get_request_img(tmp_dir, image_id, image_url, image_file, image_base64)

        # 读取图像
        img = None
        if file_path is None:
            r["error_message"] = "没有传入图片"
            return r
        else:
            img = cv2.imread(file_path)
            r["image_id"] = image_id

        # 缩放因子
        factor_mini = float(factor_mini)
        if factor_mini > 1.0 or factor_mini < 0.1:
            factor_mini = Setup.s3_factor_mini
        if factor_mini < 1.0:
            # 缩小图像加快检测速度
            width, height = img.shape[1], img.shape[0]
            img = cv2.resize(img, (int(width * factor_mini), int(height * factor_mini)))

        # 人脸检测
        total_boxes, points = self.data_processor.face_detector.detect_face(img)
        total_boxes = np.divide(total_boxes, factor_mini)
        points = np.divide(points, factor_mini)
        #print(factor_mini)
        #print(total_boxes)
        #print(points)
        faces = []
        if len(total_boxes) > 0:
            for i, box in enumerate(total_boxes):
                # 人脸矩形框的位置
                face_rectangle = {}
                face_rectangle["left"] = int(box[0])
                face_rectangle["top"] = int(box[1])
                face_rectangle["width"] = int(box[2] - box[0])
                face_rectangle["height"] = int(box[3] - box[1])

                face = {}
                face["face_token"] = id_generator()
                face["face_rectangle"] = face_rectangle

                if return_landmark == "1":
                    # 人脸的关键点坐标数组
                    landmark = []
                    for j in range(int(len(points[i]) / 2)):
                        landmark.append([int(points[i][j]),
                                         int(points[i][j + 5])])
                    face["landmark"] = landmark

                faces.append(face)
        r["faces"] = faces

        # 耗时
        times = (dt.datetime.now() - start_time)
        time_used = times.seconds * np.power(10, 3) + times.microseconds / np.power(10, 3)
        r["time_used"] = time_used

        return r

    def collect(self, cid, pid, uid, fs, ct="1", imgs=None, filenames=None):
        """
        人脸采集
        :param cid: 采集id
        :param pid: 终端id
        :param uid: 人员id
        :param fs: 人脸图文件
        :return: {'cc': 成功添加的样本数}
        """
        # 临时保存目录
        tmp_dir = Setup.s3_face_dir + "_tmp/"
        if os.path.exists(tmp_dir) == False:
            os.makedirs(tmp_dir)

        # 将文件保存到本地
        if imgs is None:
            imgs, filenames = [], []
            for f in fs:
                filename = id_generator() + ".jpg"
                file_path = os.path.join(tmp_dir, filename)
                f.save(file_path)
                imgs.append(cv2.imread(file_path))
                filenames.append(filename)

        if ct == "1":
            # 数据清洗
            imgs, filenames = self.data_processor.data_wash(imgs, filenames)
            # 数据预处理
            imgs = self.data_processor.data_preprocess(imgs)

        # 过滤跟数据集里相似度过高的图像文件
        imgs, filenames = self.data_processor.filter_sim_ds(uid, imgs, filenames)

        # 保存图像到数据集
        self.save_to_ds(cid, pid, uid, ct, imgs, filenames)

        info("新增人脸样本(uid: {}): {} 份.".format(uid, len(filenames)))

        json_r = {}
        json_r["cid"] = cid
        json_r["cc"] = len(filenames)

        return json_r

    def save_to_ds(self, cid, pid, uid, ct, imgs, filenames):
        """
        保存图像到数据集
        """
        # 记录数据库
        table_dict = TableDict()
        dict = {"id": cid, "pid": pid, "uid": uid, "ct": ct, "dt": now_dt_str(), "isdeleted": "0"}
        detail_dict_list = []

        dir = "{}/{}".format(Setup.s3_face_dir, uid)
        if os.path.exists(dir) == False:
            os.makedirs(dir)

        for i, img in enumerate(imgs):
            # 保存到数据集
            cv2.imwrite(os.path.join(dir, filenames[i]), img)

            detail_dict = {"id": filenames[i].split(".")[0], "cid": cid, "pid": pid, "uid": uid,
                           "iid": filenames[i].split(".")[0], "dt": now_dt_str()}
            detail_dict_list.append(detail_dict)
        if len(detail_dict_list) > 0:
            table_dict.save("face_collection", dict)
            table_dict.batch_save("face_collection_detail", detail_dict_list)

    def collectX(self, fs):
        """
        不间断地采集人脸
        :param fs: 人脸图文件
        :return: {'cc': 成功添加的样本数}
        """
        # 临时保存目录
        tmp_dir = Setup.s3_face_dir + "_ttt/"
        if os.path.exists(tmp_dir) == False:
            os.makedirs(tmp_dir)

        # 将文件保存到本地
        imgs, filenames = [], []
        for f in fs:
            filename = id_generator() + ".jpg"
            file_path = os.path.join(tmp_dir, filename)
            f.save(file_path)
            #imgs.append(cv2.imread(file_path))
            #filenames.append(filename)

        # 数据清洗
        #imgs, filenames = self.data_processor.data_wash(imgs, filenames)
        # 数据预处理
        #imgs = self.data_processor.data_preprocess(imgs)

        json_r = {}
        json_r["cid"] = now_dt_str()
        #json_r["cc"] = len(filenames)

        return json_r

    def load_capture_img(self, hm):
        """
        加载实时抓拍图像地址
        """
        # 临时保存目录
        face_dir = Setup.s3_face_dir + "_tmp/"
        if os.path.exists(face_dir) == False:
            return []

        lFile = list(Path(face_dir).glob("*.jpg"))
        img_urls = [str(img_url).replace("\\", "/")[2:] for img_url in lFile]
        # 倒序，新的排前面
        img_urls.sort(reverse=True)

        return img_urls[0:hm]

    def load_face_stat(self):
        """
        统计样本信息
        """
        table_dict = TableDict()

        # 统计采集人数
        sql = "select count(1) as cc from"
        sql += " (select distinct fc.uid from face_collection fc, face_collection_users fcu"
        sql += " where fc.isdeleted = '0' and fcu.isdeleted = '0' and fc.uid = fcu.uid) f"

        stat_list = table_dict.list(sql)
        cc1 =  stat_list[0]["cc"] if stat_list is not None and len(stat_list) > 0 else 0

        # 统计采集样本数
        sql = "select count(1) as cc from"
        sql += " (select fc.uid from face_collection fc, face_collection_detail fcd"
        sql += " where fc.isdeleted = '0' and fc.uid = fcd.uid) f"

        stat_list = table_dict.list(sql)
        cc2 = stat_list[0]["cc"] if stat_list is not None and len(stat_list) > 0 else 0

        json_r = {}
        json_r["cc1"] = int(cc1)
        json_r["cc2"] = int(cc2)

        return json_r

    def load_face_list1(self):
        """
        加载人脸列表，一个人对应一份记录
        """
        table_dict = TableDict()

        # 列出已采集的人，最近采集的排在前面
        sql = "select distinct fc.uid, fcu.un from face_collection fc, face_collection_users fcu"
        sql += " where fc.isdeleted = '0' and fcu.isdeleted = '0' and fc.uid = fcu.uid"
        sql += " order by fc.dt desc"

        u_list = table_dict.list(sql)

        # 查询每个人的最新采集时间以及图像
        for u in u_list:
            u_newest = self.get_newest_by_uid(u["uid"])
            if len(u_newest) > 0:
                # 附加最新采集时间以及图像
                u["dt"] = str(u_newest["dt"])
                img_path = u["uid"] + "/" + u_newest["iid"] + ".jpg"
                u["img_url"] = Setup.s3_face_dir[6:] + "/" + img_path

        return u_list

    def load_face_list2(self, uid):
        """
        加载人脸列表，列出某个人所有记录
        """
        table_dict = TableDict()

        sql = "select * from face_collection_detail"
        sql += " where uid = %(uid)s order by dt desc"

        u_list = table_dict.list(sql, {"uid": uid})
        for u in u_list:
            u["dt"] = str(u["dt"])
            img_path = u["uid"] + "/" + u["iid"] + ".jpg"
            u["img_url"] = Setup.s3_face_dir[6:] + "/" + img_path

        return u_list

    def load_face_list3(self, cid):
        """
        列出某次采集的人脸信息
        """
        table_dict = TableDict()

        sql = "select * from face_collection_detail where cid = %(cid)s"

        u_list = table_dict.list(sql, {"cid": cid})
        for u in u_list:
            u["dt"] = str(u["dt"])
            img_path = u["uid"] + "/" + u["iid"] + ".jpg"
            u["img_url"] = Setup.s3_face_dir[6:] + "/" + img_path

        return u_list

    def get_newest_by_uid(self, uid):
        """
        获取每个人的最新采集信息
        """
        table_dict = TableDict()
        sql = "select * from face_collection_detail"
        sql += " where id = (select max(id) from face_collection_detail where uid = %(uid)s)"
        u_newest_list = table_dict.list(sql, {"uid": uid})
        return {} if len(u_newest_list) == 0 else u_newest_list[0]

    def list_users_by_un_or_loginid(self, un, loginid):
        """
        根据用户名或账号模糊查找用户
        """
        table_dict = TableDict()
        sql = "select * from face_collection_users where isdeleted = '0'"
        sql1 = ""
        sql2 = ""
        dict = {}
        if un is not None:
            sql1 = "un like %(un)s"
            dict["un"] = "%{}%".format(un)
        if loginid is not None:
            sql2 = "loginid like %(loginid)s"
            dict["loginid"] = "%{}%".format(loginid)
        if sql1 != "" and sql2 != "":
            sql += " and ({} or {})".format(sql1, sql2)
        elif sql1 != "" and sql2 == "":
            sql += " and {}".format(sql1)
        elif sql1 == "" and sql2 != "":
            sql += " and {}".format(sql2)
        else:
            return []
        u_list = table_dict.list(sql, dict)
        return u_list
