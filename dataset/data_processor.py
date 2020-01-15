import numpy as np
import cv2
from dataset.mtcnn.mtcnn import Mtcnn as FaceDetector
from main.setup import Setup
from utils.utils_cv import distance_l2_relative as distance, max_box, list_imgs_files


class DataProcessor:
    """
    数据处理器
    """

    def __init__(self):
        # 人脸检测器
        self.face_detector = FaceDetector()

    def filter_sim_ds(self, uid, imgs, filenames):
        """
        过滤跟数据集里相似度过高的图像文件
        """
        imgs_filtered, filenames_filtered = [], []

        dir = "{}/{}".format(Setup.s3_face_dir, uid)
        imgs_e, _ = list_imgs_files(dir)
        for i, img in enumerate(imgs):
            is_sim = False
            for img_e in imgs_e:
                if distance(img, img_e) < Setup.s3_distance_t:
                    is_sim = True
                    break
            if is_sim == False:
                imgs_filtered.append(img)
                filenames_filtered.append(filenames[i])

        return imgs_filtered, filenames_filtered

    def data_preprocess(self, imgs):
        """
        数据预处理
        """
        imgs_pp = []

        for img in imgs:
            # 统一尺寸
            img_pp = cv2.resize(img, (Setup.s3_size, Setup.s3_size))
            imgs_pp.append(img_pp)

        return imgs_pp

    def data_wash(self, imgs, filenames=None):
        """
        数据清洗-追加人脸检测
        """
        imgs_washed, filenames_washed = [], []

        # 用不同的人脸检测法再检测多次，确保采集的是人脸，并确定人脸区域
        for i, img in enumerate(imgs):
            width, height = img.shape[1], img.shape[0]

            # 缩放因子
            factor_mini = Setup.s3_factor_mini
            # 缩小图像加快检测速度
            img_mini = cv2.resize(img, (int(width * factor_mini), int(height * factor_mini)))

            # 人脸检测
            total_boxes, points = self.face_detector.detect_face(img_mini)
            if len(total_boxes) == 0:
                continue

            # 只要最大的图
            box_max = max_box(total_boxes)
            p1 = (int(np.max([0, box_max[0]]) / factor_mini), int(box_max[1] / factor_mini))
            p2 = (int(np.max([0, box_max[2]]) / factor_mini), int(box_max[3] / factor_mini))

            # 提取人脸图像
            img_face = img[p1[1]: p2[1], p1[0]: p2[0], :]

            imgs_washed.append(img_face)
            if filenames is not None:
                filenames_washed.append(filenames[i])

        return imgs_washed, filenames_washed
