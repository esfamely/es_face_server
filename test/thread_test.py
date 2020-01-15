from time import sleep
import threading

thread_lock = threading.Lock()


def ko1():
    thread_lock.acquire()

    for i in range(5):
        print("love szy ......")
        sleep(1.5)
        print("miss szy ......")

    thread_lock.release()


def ko2():
    thread_lock.acquire()

    for i in range(3):
        print("love hd.")
        sleep(1.5)
        print("miss hd.")

    thread_lock.release()


def ko():
    #ko1()
    #ko2()
    t1 = threading.Thread(target=ko1, daemon=False)
    t1.start()
    t2 = threading.Thread(target=ko2, daemon=False)
    t2.start()
    print("ko")


ko()
