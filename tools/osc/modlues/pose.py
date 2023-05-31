import mediapipe
import numpy as np
import cv2

class Pose:
    
    def __init__(self):
      self.mp_pose = mediapipe.solutions.pose

    def estimate(self, frame: np.ndarray) -> np.ndarray:
      with self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)
        if results.pose_landmarks is None:
          return None
        return results.pose_landmarks
      
    def estimate_A(self, frame: np.ndarray) -> np.ndarray:
      frame = self.estimate(frame)
      if frame is None:
        return None
      frame = self.nose_centered(frame)
      frame = self.filter_landmark(frame)
      return frame

    @staticmethod
    def filter_landmark(landmarks: list) -> list:
      l1 = [ _*2 for _ in [0,11,12,13,14,15,16,23,24]]
      l2 = [ _ + 1 for _ in l1]
      filter_landmarks = [landmarks[y] for x in zip(l1, l2) for y in x]
      return filter_landmarks

    @staticmethod
    def nose_centered(landmarks: list):
      landmark_point = []
      # ランドマークの中心をNoseに設定する
      center_x = landmarks.landmark[0].x
      center_y = landmarks.landmark[0].y

      for landmark in landmarks.landmark:
        landmark_x = landmark.x - center_x
        landmark_y = landmark.y - center_y
        landmark_point.extend([landmark_x, landmark_y])
      return landmark_point