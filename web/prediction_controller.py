# 防止json.dumps中文乱码
from __future__ import unicode_literals
import json
from service.main_service import app, prediction_service
from utils.utils_web import getParameter, getFile


class PredictionController:
    @app.route("/face/prediction/compare", methods=["POST"])
    def compare():
        """
        1 vs 1 人脸比对
        """
        image_url1 = getParameter("image_url1")
        image_file1 = getFile("image_file1")
        image_base64_1 = getFile("image_base64_1")
        face_rectangle1 = getParameter("face_rectangle1")
        image_url2 = getParameter("image_url2")
        image_file2 = getFile("image_file2")
        image_base64_2 = getFile("image_base64_2")
        face_rectangle2 = getParameter("face_rectangle2")

        r = prediction_service.compare(image_url1, image_file1, image_base64_1, face_rectangle1,
                                       image_url2, image_file2, image_base64_2, face_rectangle2)

        return json.dumps(r)

    @app.route("/face/prediction/search", methods=["POST"])
    def search():
        """
        1 vs n 人脸检索
        """
        image_url = getParameter("image_url")
        image_file = getFile("image_file")
        image_base64 = getFile("image_base64")
        face_rectangle = getParameter("face_rectangle")

        r = prediction_service.search(image_url, image_file, image_base64, face_rectangle)

        return json.dumps(r)

    @app.route("/face/prediction/load_recent_prediction", methods=["GET", "POST"])
    def load_recent_prediction():
        """
        加载最近的识别结果
        """
        # 要取多少记录
        hm = int(getParameter("hm", 3))
        json_r = prediction_service.load_recent_prediction(hm)
        return json.dumps(json_r, ensure_ascii=False)
