# -*-coding:utf-8-*-

import cv2
import utils.utils_sys as utils_sys
import utils.utils_cv as utils_cv

# 运行参数
# 缩小比例
s_factor = 0.25
# 最大帧数
index_max = 100000
# 每隔几帧进行一次人脸检测
s_ffd = 5
# 人脸边界预留宽度
s_border = 30
# 上一次检测到的人脸区域
last_face = None
# esc键编码
keycode_esc = 27

# 加载人脸检测器
detector = cv2.CascadeClassifier("../cv/haarcascade/haarcascade_frontalface_default.xml")

# 开启摄像头
camera = cv2.VideoCapture(0)
index = 0
while True:
    grabbed, frame = camera.read()
    height, width = frame.shape[0:2]

    # 照镜效果
    frame = cv2.flip(frame, 1)

    face = None
    # 为了提高fps，并非每帧都进行人脸检测，不进行检测的帧取上一次的检测结果
    if index % s_ffd == 0:
        # 缩小图像加快检测速度
        frame_mini = cv2.resize(frame, (int(width * s_factor), int(height * s_factor)))
        # 实时人脸检测
        faces = detector.detectMultiScale(frame_mini,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(5, 5),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) == 0:
            face = None
        else:
            # 只要最大的图
            face = utils_cv.max_rect(faces)
        last_face = face
    else:
        face = last_face

    if face is not None:
        # 人脸标记框左上点与右下点
        p1 = (int(face[0] / s_factor) - s_border, int(face[1] / s_factor) - s_border)
        p2 = (int((face[0] + face[2]) / s_factor) + s_border, int((face[1] + face[3]) / s_factor) + s_border)
        # 越界的图不要
        if p1[0] > 0 and p1[1] > 0 and p2[0] < width and p2[1] < height:
            # 保存并显示人脸
            face_img = frame[p1[1]:p2[1], p1[0]:p2[0] :]
            cv2.imwrite("face_ttt/{}.jpg".format(utils_sys.id_generator()), face_img)
            cv2.rectangle(frame, p1, p2, [255, 255, 255], 5)

    # 显示图像
    cv2.imshow("es_face", frame)

    # 按esc退出
    if cv2.waitKey(1) == keycode_esc:
        break

    index = (index + 1) % index_max

camera.release()
cv2.destroyAllWindows()
