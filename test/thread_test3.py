import threading
from utils.utils_sys import info
import cv2
from main.setup import Setup
from time import sleep

thread_lock = threading.RLock()


class FaceTest0:
    def __init__(self, path):
        self.detector = cv2.CascadeClassifier(Setup.s1_cascade_path)
        self.path = path

    def test1(self):
        thread_lock.acquire()
        info("ko1")
        img1 = cv2.imread(self.path)
        rects1 = self.detector.detectMultiScale(img1,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        print(rects1)
        thread_lock.release()

    def test2(self):
        thread_lock.acquire()
        info("ko2")
        img2 = cv2.imread("D:/s5/lena/103752.jpg")
        rects2 = self.detector.detectMultiScale(img2,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        print(rects2)
        thread_lock.release()


def ko():
    for i in range(5):
        path = "D:/s5/lena/103708.jpg"
        if i % 2 == 0:
            path = "D:/s5/lena/lena.png"
        faceTest0 = FaceTest0(path)

        for j in range(50):
            print(j)
            t1 = threading.Thread(target=faceTest0.test1, daemon=False)
            t1.start()
            t2 = threading.Thread(target=faceTest0.test2, daemon=False)
            t2.start()

            sleep(2.5)


ko()
