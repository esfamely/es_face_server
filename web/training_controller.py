import json
from service.main_service import app, training_service


class TrainingController:
    @app.route("/face/training/train", methods=["GET", "POST"])
    def train():
        """
        模型训练
        """
        training_service.train()

        dict = {}
        dict["result"] = ""

        return json.dumps(dict)

    @app.route("/face/training/get_train_info", methods=["GET", "POST"])
    def get_train_info():
        """
        得到最近一次训练的情况
        """
        r = training_service.get_train_info()

        return json.dumps(r)
