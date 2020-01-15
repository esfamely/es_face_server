import numpy as np
import cv2
from collection.haarcascade.haar_cascade import HaarCascade

haar_cascade = HaarCascade()

img = cv2.imread("D:/s5/lena/lena.png")
rects = haar_cascade.detect_face(img)


def area(rect):
    return rect[2] * rect[3]


list = [area(rect) for rect in rects]
max_index = np.argmax(list)

for i, rect in enumerate(rects):
    print(rect)
    p1 = (rect[0], rect[1])
    p2 = (rect[0] + rect[2], rect[1] + rect[3])
    img = cv2.rectangle(img, p1, p2, (255, 0, 0), 3)

'''rect = rects[max_index]
p1 = (rect[0], rect[1])
p2 = (rect[0] + rect[2], rect[1] + rect[3])
img = cv2.rectangle(img, p1, p2, (255, 0, 0), 3)'''

cv2.imshow("es", img)
cv2.waitKey(0)
