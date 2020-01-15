class Setup:
    """
    系统运行参数
    """

    @staticmethod
    def export(startswith="s"):
        dict = {}
        for item in Setup.__dict__.items():
            if str(item[0]).startswith(startswith):
                dict[item[0]] = item[1]
        return dict

    """
    系统运行平台
    """
    # 数据库IP
    s0_db_ip = "192.168.1.141"
    # 数据库名
    s0_db_name = "es_face"
    # 数据库账号
    s0_db_login = "es"
    # 数据库密码
    s0_db_pw = "123456"

    """
    人脸采集器
    """
    # 人脸检测haar cascade xml路径
    s1_cascade_path = "../static/opencv/haarcascade_frontalface_default.xml"
    # 缩放因子
    s1_factor_mini = 0.25
    # 人脸边界预留宽度
    s1_border = 50
    # 人脸图像统一尺寸
    s1_size = 160
    # 每隔几帧保存一次人脸图像
    s1_frame_cc = 3
    # 保存满几张就提交
    s1_submit_cc = 3
    # 图像相似度阀值
    s1_distance_t = 63.0

    """
    数据处理器
    """
    # 人脸图像统一尺寸
    s3_size = 160
    # 人脸数据存放路径
    s3_face_dir = "../web/static/dataset/face_" + str(s3_size)
    # 缩放因子
    s3_factor_mini = 0.5
    # 最小检测尺寸
    s3_minsize = 10
    # 三个cnn的检测阀值
    s3_threshold = [0.6, 0.7, 0.7]
    # 尺度因子
    s3_factor = 0.709
    # 图像相似度阀值
    s3_distance_t = 63.0

    """
    识别模型
    """
    # 是否使用特征提取法
    s4_use_feature_extract = 1
    # 特征文件存放路径
    s4_feature_dir = "../dataset/face_{}_feature".format(s3_size)
    # Facenet特征提取器已训练模型路径
    s4_facenet_model_path = "../../facenet/models/20180402-114759/20180402-114759.pb"
    # Facenet特征提取每批大小
    s4_facenet_batch_size = 16
    # 距离阀值
    s4_distance_threshold = 1.05
