import modlues

import cv2

def main():
    pose_process_test()

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

if __name__ == '__main__':
    main()