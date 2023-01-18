import emotion_input
import pose_input
import cv2
from datetime import datetime
import csv_output

class DataService:
  def __init__(self):
    self.csv_writer = csv_output.CsvSaveService("output.csv")

  def run(self):
    data_source = dict()
    data_source["pose"] = None
    data_source["emotion"] = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    while cap.isOpened():
      ret, frame = cap.read()
      if ret is False:
        break
      pose_data = pose_input.get_pose(frame)

      # ディスプレイ表示
      # cv2.imshow('chapter02', frame)

      key = cv2.waitKey(1)
      if key == 27:  # ESC
        break
      
      data_source["pose"] = pose_data
      data_source["timestamp"] = datetime.now()

      self.csv_writer.save_data(data_source)


if __name__ == "__main__":
  test = DataService()
  test.run()