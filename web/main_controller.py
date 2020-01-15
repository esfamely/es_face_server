import json
from flask import render_template, send_from_directory
from service.main_service import app, MainService

main_service = MainService()


class MainController:
    @app.route('/face', methods=['GET'])
    def index():
        """
        主页
        """
        # 当前访问者是终端机则访问采集页，一般访问者访问主页
        if main_service.is_camera():
            return render_template("collection/index.html")
        else:
            return render_template("index.html")

    @app.route('/favicon.ico', methods=['GET'])
    def favicon():
        """
        站点图标
        """
        return send_from_directory('../web/templates',
                                   'favicon.ico', mimetype='image/vnd.microsoft.icon')

    @app.route('/face/camera', methods=['GET'])
    def camera():
        """
        采集机视频页
        """
        return render_template("collection/camera.html")

    @staticmethod
    @app.route('/face/remote/<camera_sn>', methods=['GET'])
    def remote(camera_sn):
        """
        远程视频监控页面
        """
        #print(camera_sn)
        return render_template("webrtc/client.html", camera_sn=camera_sn)

    @app.route("/face/load_setup", methods=["POST"])
    def load_setup():
        """
        读取系统配置
        """
        json_setup = main_service.load_setup()
        return json.dumps(json_setup)

    @app.route('/face/test_api', methods=['GET'])
    def test_api():
        return render_template("test/test_api.html")

    @app.route('/face/test_detect', methods=['GET'])
    def test_detect():
        return render_template("test/test_detect.html")

    @app.route('/face/test_compare', methods=['GET'])
    def test_compare():
        return render_template("test/test_compare.html")

    @app.route('/face/test_search', methods=['GET'])
    def test_search():
        return render_template("test/test_search.html")
