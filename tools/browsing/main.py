from flask import Flask, Response, render_template
from flask_socketio import SocketIO
from collections import deque
import cv2
import mediapipe as mp
import nnabla as nn
from nnabla.utils.nnp_graph import NnpLoader
import numpy as np

# initialize 
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# socketioとmediapipeの対応付けリスト
selectParts = {"LEFT_SHOULDER" : mp_pose.PoseLandmark.LEFT_SHOULDER, 
              "RIGHT_SHOULDER" : mp_pose.PoseLandmark.RIGHT_SHOULDER,
              "LEFT_ELBOW" : mp_pose.PoseLandmark.LEFT_ELBOW,
              "RIGHT_ELBOW" : mp_pose.PoseLandmark.RIGHT_ELBOW,
              "LEFT_WRIST" : mp_pose.PoseLandmark.LEFT_WRIST,
              "RIGHT_WRIST" : mp_pose.PoseLandmark.RIGHT_WRIST,
              "LEFT_HIP" : mp_pose.PoseLandmark.LEFT_HIP,
              "RIGHT_HIP" : mp_pose.PoseLandmark.RIGHT_HIP}
# ランドマーク初期値
graph_landmark = mp_pose.PoseLandmark.LEFT_WRIST

# 推論設定
results = None # 推論結果
frame_num = 150 # 推論フレーム
pose_storage = deque(maxlen=frame_num) # 推論フレームのストレージ

# nnablaの初期設定
batch_size = 1
nnp = NnpLoader("18landmark.nnp")
net = nnp.get_network("Runtime", batch_size=batch_size)
x = net.inputs["Input"]
y = net.outputs["y'"]

# 推論結果を鼻からの相対距離にする
def nose_centered(landmarks):
    landmark_point = []
    # ランドマークの中心をNoseに設定する
    center_x = landmarks.landmark[0].x
    center_y = landmarks.landmark[0].y

    for landmark in landmarks.landmark:
        landmark_x = landmark.x - center_x
        landmark_y = landmark.y - center_y
        landmark_point.extend([landmark_x, landmark_y])
    return landmark_point


def gen_frames():  
  global graph_landmark
  cap = cv2.VideoCapture(0)  # Capture video from webcam
  
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      ret, frame = cap.read()  # read the frame

      if not ret:
        # cant't receive frame
        break

      # ポーズ処理

      # Flip the image horizontally for a selfie-view display
      frame = cv2.flip(frame, 1)
      # Convert the BGR image to RGB
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False

      # Make pose detection
      results = pose.process(image)

      # Draw the pose annotations on the image
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

      # 推論

      if results.pose_landmarks is not None:
        # landmark = results.pose_landmarks.landmark[graph_landmark]
        # pre_process landmark
        landmarks = nose_centered(results.pose_landmarks)
        # select only 18 landmarks
        l1 = [ _*2 for _ in [0,11,12,13,14,15,16,23,24]]
        l2 = [ _ + 1 for _ in l1]
        filter_landmarks = [landmarks[y] for x in zip(l1, l2) for y in x]

        # store frame for estimation
        pose_storage.append(filter_landmarks)

        # store send data
        landmark = results.pose_landmarks.landmark[graph_landmark]

        if (len(pose_storage) >= frame_num): 
          # estimation
          x.d = np.array(pose_storage, dtype=np.float16)
          y.forward()
          ans = np.array(y.d.copy())
          ans = ans.flatten()
          socketio.emit('newcoords', {'x': landmark.x, 'y': landmark.y, 'data': ans.tolist() }) 
        else:
          socketio.emit('newcoords', {'x': landmark.x, 'y': landmark.y, 'data': [0, 0, 0, 0, 0]})
      else:
        # store frame for estimation (if no pose detected, store 0)
        pose_storage.append([0]*18)
      

      # Convert the image color and return as a video frame
      ret, buffer = cv2.imencode('.jpg', image)
      frame = buffer.tobytes()
      yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def index():
    return render_template("index.html")

# グラフのランドマークを変更する
@socketio.on('selectPart')
def handle_part_select(data):
  global graph_landmark
  graph_landmark = selectParts[data['part']]

if __name__ == '__main__':
    app.run(debug=True)
