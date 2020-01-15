import cv2
import tensorflow as tf
from dataset.mtcnn.detect_face import create_mtcnn, detect_face
from main.setup import Setup
from utils.utils_cv import max_box, add_border

sess = tf.Session()
with sess.as_default():
    pnet, rnet, onet = create_mtcnn(sess, None)

img = cv2.imread("D:/s5/lena/103752.jpg")
img = add_border(img)

for i in range(15):
    total_boxes, _ = detect_face(img, Setup.s3_minsize, pnet, rnet, onet,
                                      Setup.s3_threshold, Setup.s3_factor)
    print(total_boxes)
    if len(total_boxes) > 0:
        box = max_box(total_boxes)

        img = img[int(box[1]): int(box[3]), int(box[0]): int(box[2]), :]
        img = add_border(img)

        cv2.imshow("es", img)
        cv2.waitKey(0)
