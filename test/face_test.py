import numpy as np
import cv2
import tensorflow as tf
from dataset.mtcnn.detect_face import create_mtcnn, detect_face
from main.setup import Setup

sess = tf.Session()
with sess.as_default():
    pnet, rnet, onet = create_mtcnn(sess, None)

img = cv2.imread("D:/s5/lena/s05.jpg")

# 缩放因子
factor_mini = Setup.s3_factor_mini
# 缩小图像加快检测速度
width, height = img.shape[1], img.shape[0]
img_mini = cv2.resize(img, (int(width * factor_mini), int(height * factor_mini)))

total_boxes, points = detect_face(img_mini, Setup.s3_minsize, pnet, rnet, onet,
                                  Setup.s3_threshold, Setup.s3_factor)
total_boxes = np.divide(total_boxes, factor_mini)
points = np.transpose(np.divide(points, factor_mini))
print(total_boxes)
print(points)

for i, box in enumerate(total_boxes):
    p1 = (int(box[0]), int(box[1]))
    p2 = (int(box[2]), int(box[3]))
    img = cv2.rectangle(img, p1, p2, (255, 0, 0), 3)

    color = (255, 0, 0) if i % 2 == 0 else (0, 255, 0)
    for j in range(int(len(points[i]) / 2)):
        img = cv2.circle(img, (int(points[i][j]), int(points[i][j + 5])),
                         5, color, thickness=3)

cv2.imshow("es", img)
cv2.waitKey(0)
