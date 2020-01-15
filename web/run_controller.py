from service.main_service import app
from web.collection_controller import CollectionController
from web.main_controller import MainController
from web.prediction_controller import PredictionController
from web.training_controller import TrainingController

if __name__ == "__main__":
    MainController()
    CollectionController()
    TrainingController()
    PredictionController()

    #app.run(host="127.0.0.1", port=7036, debug=True)
    #app.run(host="192.168.1.36", port=7036, debug=True)
    # 启用https
    app.run(host="192.168.1.141", port=7036, debug=True,
            ssl_context=('templates/ssl/ssl.crt', 'templates/ssl/ssl.key'))
