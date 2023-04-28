# import 
import os
import cv2

# user file
import landmark
import csv_output

print("processing...")
landmarker = landmark.LandmarkCreator()

# videoディレクトリ以下の動画ファイルをcsvに変換する
video_directory = os.path.dirname(__file__) + "./video"
file_list = os.listdir(video_directory)
entire_frame = 150 # 全フレーム (5s x 30fps)

for file in file_list:
    print("file: " + file + " is processing...")
    video_capture = cv2.VideoCapture(video_directory + "/" + file)
    csv_save = csv_output.CsvSaveService(os.path.dirname(__file__) + "/" + file + ".csv")

    # フレームカウントの初期化
    frame_count = 0

    while video_capture.isOpened():
      if frame_count >= entire_frame:
        # 指定フレームより超えると終了
        break
      
      # 1フレームの読み取り
      ret, frame = video_capture.read()

      if ret is False:
        # フレーム読み込み失敗時は終了
        break

      # 色空間の調整
      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      landmarker.process(rgb_frame)
      data_source = {}
      data_source["pose"]  = landmarker.get_landmarks()

      csv_save.save_data(data_source)

      frame_count += 1

print("ended")