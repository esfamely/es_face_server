import threading
from utils.utils_sys import info
import cv2
from main.setup import Setup
from time import sleep

thread_lock = threading.RLock()
img_path = None


class FaceTest0:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(Setup.s1_cascade_path)

    def test1(self):
        img1 = cv2.imread("D:/s5/lena/103708.jpg")
        rects1 = self.detector.detectMultiScale(img1,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        print(rects1)

    def test2(self):
        img2 = cv2.imread("D:/s5/lena/lena.png")
        rects2 = self.detector.detectMultiScale(img2,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        print(rects2)


class FaceTest1(threading.Thread):
    def __init__(self, name, img_path):
        threading.Thread.__init__(self)
        self.daemon = False
        self.name = name
        self.img_path = img_path
        self.detector = cv2.CascadeClassifier(Setup.s1_cascade_path)

    def run(self):
        thread_lock.acquire()
        info(self.img_path)
        img = cv2.imread(self.img_path)
        rects = self.detector.detectMultiScale(img,
                                               scaleFactor=1.1,
                                               minNeighbors=5,
                                               minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        print(rects)
        thread_lock.release()


class FaceTest2(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.daemon = False
        self.name = name
        self.detector = cv2.CascadeClassifier(Setup.s1_cascade_path)

    def run(self):
        while True:
            global img_path
            if img_path is None:
                continue

            info(img_path)
            img = cv2.imread(img_path)
            rects = self.detector.detectMultiScale(img,
                                                   scaleFactor=1.1,
                                                   minNeighbors=5,
                                                   minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            print(rects)

            img_path = None


faceTest0 = FaceTest0()


def ko1():
    thread_lock.acquire()

    info("ko1")
    faceTest0.test1()

    thread_lock.release()


def ko2():
    thread_lock.acquire()

    info("ko2")
    faceTest0.test2()

    thread_lock.release()


def ko():
    index = 0
    while True:
        print(index)

        t1 = threading.Thread(target=ko1, name="est1-{:03d}".format(index), daemon=False)
        t1.start()
        t2 = threading.Thread(target=ko2, name="est2-{:03d}".format(index), daemon=False)
        t2.start()

        '''t1 = FaceTest1("est1-{:03d}".format(index), "D:/s5/lena/103708.jpg")
        t1.start()
        t2 = FaceTest1("est2-{:03d}".format(index), "D:/s5/lena/lena.png")
        t2.start()'''

        index += 1
        sleep(2.5)

        '''faceTest0.test1()
        faceTest0.test2()'''


def ko0():
    t = FaceTest2("est")
    t.start()
    print("ko")

    index = 0
    while True:
        global img_path

        print(index)

        if index % 2 == 0:
            img_path = "D:/s5/lena/103708.jpg"
        else:
            img_path = "D:/s5/lena/lena.png"

        index += 1
        sleep(2.5)


ko()
