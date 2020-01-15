from flask import Flask, request
from service.collection_service import CollectionService
from service.camera_service import CameraService
from service.prediction_service import PredictionService
from main.setup import Setup
from service.training_service import TrainingService

app = Flask(__name__, template_folder="../web/templates", static_folder="../web/static")

collection_service = CollectionService()
camera_service = CameraService()
training_service = TrainingService()
prediction_service = PredictionService(collection_service, training_service)


class MainService:
    """
    全局功能
    """

    def load_setup(self, startswith="s1"):
        """
        读取系统配置
        """
        setup = Setup.export(startswith=startswith)
        # 追加当前访问终端机序号
        setup["camera_sn"] = self.get_camera_sn(request.remote_addr)

        return setup

    def is_camera(self):
        """
        当前访问者是否终端机
        """
        ip = request.remote_addr
        camera = camera_service.get_by_ip(ip)
        return False if camera is None else True

    def get_camera_sn(self, ip):
        """
        读取当前访问终端机序号
        """
        camera = camera_service.get_by_ip(ip)
        return "" if camera is None else camera["sn"]
