import cv2
import mediapipe as mp


mp_pose = mp.solutions.pose

# MediaPipe pose初期化
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_pose(frame):
    
    # MediaPipeで扱う画像は、OpenCVのBGRの並びではなくRGBのため変換
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 画像をリードオンリーにしてpose検出処理実施
    rgb_image.flags.writeable = False
    #姿勢
    pose_results = pose.process(rgb_image)
    rgb_image.flags.writeable = True

    result = {}
    if pose_results.pose_landmarks:
        for i,landmark in enumerate(pose_results.pose_landmarks.landmark):
            result[mp_pose.PoseLandmark(i).name] = {"x":landmark.x,"y":landmark.y}
    return result


if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        
        result = get_pose(frame)
        
        
        # ディスプレイ表示
        cv2.imshow('chapter02', frame)
        
        #print(pose_results)
        # キー入力(ESC:プログラム終了)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
    print(result)
    # リソースの解放
    cap.release()
    pose.Close()
    cv2.destroyAllWindows()