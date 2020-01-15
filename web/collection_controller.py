# 防止json.dumps中文乱码
from __future__ import unicode_literals
import json
from flask import request
from service.main_service import app, collection_service, camera_service, prediction_service
from utils.utils_sys import id_generator, now_dt_str
from utils.utils_web import getParameter, getFile
from main.setup import Setup


class CollectionController:
    @app.route("/face/collection/detect", methods=["POST"])
    def detect():
        """
        人脸检测
        """
        image_url = getParameter("image_url")
        image_file = getFile("image_file")
        image_base64 = getFile("image_base64")
        return_landmark = getParameter("return_landmark", "1")
        factor_mini = getParameter("factor_mini", Setup.s3_factor_mini)

        r = collection_service.detect(image_url, image_file, image_base64, return_landmark, factor_mini)

        return json.dumps(r)

    @app.route("/face/collection/collect", methods=["POST"])
    def collect():
        """
        人脸采集
        """
        # 采集id
        cid = getParameter("cid", id_generator())
        # 终端id
        pid = getParameter("pid", "1")
        # 人员id
        uid = request.form["uid"]
        # 人脸图数
        img_cc = int(request.form["img_cc"])
        # 是否进行识别
        to_rec = int(request.form["to_rec"])
        # 人脸图文件
        fs = []
        for m in range(img_cc):
            f = request.files["file" + str(m)]
            fs.append(f)

        json_r = {}
        if to_rec == 0:
            json_r = collection_service.collect(cid, pid, uid, fs)
        else:
            json_r = prediction_service.predict(cid, pid, fs)

        return json.dumps(json_r)

    @app.route("/face/collection/collectX", methods=["POST"])
    def collectX():
        """
        不间断地采集人脸
        """
        # 人脸图数
        img_cc = int(request.form["img_cc"])
        # 人脸图文件
        fs = []
        for m in range(img_cc):
            f = request.files["file" + str(m)]
            fs.append(f)

        json_r = collection_service.collectX(fs)

        return json.dumps(json_r)

    @app.route("/face/collection/load_capture_img", methods=["GET", "POST"])
    def load_capture_img():
        """
        加载实时抓拍图像地址
        """
        # 要取多少图像
        hm = int(getParameter("hm", 3))
        json_r = collection_service.load_capture_img(hm)
        return json.dumps(json_r)

    @app.route("/face/collection/load_face_stat", methods=["GET", "POST"])
    def load_face_stat():
        """
        统计样本信息
        """
        json_r = collection_service.load_face_stat()
        return json.dumps(json_r)

    @app.route("/face/collection/load_face_list1", methods=["GET", "POST"])
    def load_face_list1():
        """
        加载人脸列表，一个人对应一份记录
        """
        u_list = collection_service.load_face_list1()
        return json.dumps(u_list, ensure_ascii=False)

    @app.route("/face/collection/load_face_list2", methods=["GET", "POST"])
    def load_face_list2():
        """
        加载人脸列表，列出某个人所有记录
        """
        uid = request.get_json()["uid"]
        u_list = collection_service.load_face_list2(uid)
        return json.dumps(u_list)

    @app.route("/face/collection/load_face_list3", methods=["GET", "POST"])
    def load_face_list3():
        """
        列出某次采集的人脸信息
        """
        cid = request.get_json()["cid"]
        u_list = collection_service.load_face_list3(cid)
        return json.dumps(u_list)

    @app.route("/face/collection/list_users_by_un_or_loginid", methods=["GET", "POST"])
    def list_users_by_un_or_loginid():
        """
        根据用户名或账号模糊查找用户
        """
        un = getParameter("un")
        loginid = getParameter("loginid")
        u_list = collection_service.list_users_by_un_or_loginid(un, loginid)
        return json.dumps(u_list)

    @app.route("/face/collection/list_camera", methods=["GET", "POST"])
    def list_camera():
        """
        列出全部终端机
        """
        c_list = camera_service.list_camera()
        return json.dumps(c_list)

    @app.route("/face/collection/get_camera", methods=["GET", "POST"])
    def get_camera():
        """
        取得终端机信息
        """
        id = getParameter("id")
        camera = camera_service.get_camera(id)
        return json.dumps(camera)

    @app.route("/face/collection/save_camera", methods=["POST"])
    def save_camera():
        """
        新增或修改终端机
        """
        id = getParameter("id")
        cn = getParameter("cn")
        ip = getParameter("ip")
        sn = getParameter("sn")
        cno = getParameter("cno", 1)
        tips = getParameter("tips")
        # result记录异常信息
        id, result = camera_service.save_camera(id, cn, ip, sn, cno, tips)
        return json.dumps({"id": id, "result": result})

    @app.route("/face/collection/del_camera", methods=["POST"])
    def del_camera():
        """
        删除终端机
        """
        id_list = getParameter("id_list")
        camera_service.del_camera(id_list)
        return json.dumps({"result": ""})
