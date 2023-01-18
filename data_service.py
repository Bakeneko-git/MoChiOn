import emotion_input
import pose_input

def data_servcie():
  while True:
    emotion_data = emotion_input.get_emotion()
    pose_data = pose_input.get_pose()