import modlues
import numpy as np

import cv2

def main():
    process()


def process():
    pose_process = modlues.pose_process.Pose_process("./model/model.nnp")
    osc_client = modlues.osc.OSC()
    thred = 0.98
    for res in pose_process.start():
        if res is None:
            continue
        max_prob = np.max(res)
        if (max_prob < thred):
            continue
        ans = np.argmax(res)
        osc_client.send("/pose", int(ans))


def camera_test():
    camera = modlues.camera.Camera()
    for frame in camera.capture():
        # 表示する
        cv2.imshow('frame', frame)
        # qキーが押されたら途中終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            

def image_process_test():
    image_process = modlues.image_process.Image_process()
    for frame in image_process.start():
        if frame is None:
            continue
        # 表示する
        print(frame)
    
def pose_process_test():
    pose_process = modlues.pose_process.Pose_process("./model/model.nnp")
    for frame in pose_process.start():
        if frame is None:
            continue
        # 表示する
        print(frame)

def osc_test():
    import time
    osc_client = modlues.osc.OSC()
    for i in range(10):
        osc_client.send("/test", i)
        print("sended")
        time.sleep(1)   

if __name__ == '__main__':
    main()