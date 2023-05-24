import mediapipe as mp

class LandmarkCreator:
    # initialize
    # mediaパイプを初期化する
    def __init__(self):
      # MediaPipe poses初期化
      self.mp_pose = mp.solutions.pose
      self.pose = self.mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
      )       

    # デコンストラクタ
    # mediapipeを破棄する
    def __del__(self):
      self.pose.close()

    # 画像を入力として画像のランドマークを取得する
    def process(self,image, nose_centered = True):
        pose_result = self.pose.process(image)
        if (pose_result.pose_landmarks):
            # データが保存されている
            if (nose_centered):
                # nose中心にランドマークを変換する
                self.pose_landmarks = self.nose_centered(pose_result.pose_landmarks)
            return True
        
    def get_landmarks(self):
      result = {}
      for idx, landmark in enumerate(self.pose_landmarks):
        # ランドマーク名 : [x, y] の辞書を作成
        result[self.mp_pose.PoseLandmark(idx).name] = {"x":landmark["x"],"y":landmark["y"]}
      return result
    
    def get_frip_landmarks(self):
      result = {}
      for idx, landmark in enumerate(self.pose_landmarks):
        # ランドマーク名 : [x, y] の辞書を作成
        result[self.mp_pose.PoseLandmark(idx).name] = {"x":-landmark["x"],"y":landmark["y"]}
      return result

        
    @staticmethod
    def nose_centered(landmarks):
      landmark_point = []
      # 画像サイズの取得
      
      # ランドマークの中心をNoseに設定する
      center_x = landmarks.landmark[0].x
      center_y = landmarks.landmark[0].y

      for _, landmark in enumerate(landmarks.landmark):
          landmark_x = landmark.x - center_x
          landmark_y = landmark.y - center_y
          landmark_point.append({"x":landmark_x, "y":landmark_y})
      return landmark_point
