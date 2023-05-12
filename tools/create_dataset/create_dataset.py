# chapter03.py
# -*- coding: utf-8 -*-

# import
import csv
import argparse
from collections import deque
from datetime import datetime
import os

import cv2
import mediapipe as mp

# ユーザーファイルのインポート
import csv_output

# 関数宣言

# ランドマークの画像上の位置を算出する関数
def calc_landmark_list(image, landmarks):
    landmark_point = []
    image_width, image_height = image.shape[1], image.shape[0]

    #center_x = (landmarks.landmark[23].x + landmarks.landmark[24].x) / 2
    #center_y = (landmarks.landmark[23].y + landmarks.landmark[24].y) / 2
    center_x = landmarks.landmark[0].x
    center_y = landmarks.landmark[0].y
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = landmark.x - center_x
        landmark_y = landmark.y - center_y
        landmark_point.append({"x":landmark_x, "y":landmark_y})

    return landmark_point


# 座標履歴を描画する関数
def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv2.circle(image, (point[0], point[1]), 1 + int(index / 2),(255, 0, 0), 2)
    return image


# CSVファイルに座標履歴を保存する関数
def logging_csv(gesture_id, csv_path, width, height, point_history_list):
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([gesture_id, width, height, *point_history_list])
    return

# コマンドライン引数取得
parser = argparse.ArgumentParser()
parser.add_argument("--pat", type=int, default=0) # ファイル名用
parser.add_argument("--path", type=str) # ディレクトリ動画の選択
parser.add_argument("--frames", type=int, default=150) # 取得フレーム数(30fps, 5s)
parser.add_argument("--frip", type=int, default=0) # 取得フレーム数(30fps, 5s)
args = parser.parse_args()

pat_dict = {
    0:"10000",
    1:"01000",
    2:"00100",
    3:"00010",
    4:"00001"
}

# MediaPipe poses初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# fripの有効
frip = (args.frip == 1)

# ディレクトリのパスを保存
video_directory = args.path 

# ディレクトリ内のファイルリストを取得
file_list = os.listdir(video_directory)

# すべてのファイルをCSVに保存する
for file in file_list:
    # 保存CSVファイルの作成
    csv_save = csv_output.CsvSaveService(os.path.splitext(os.path.basename(file))[0] + "_" + pat_dict[args.pat] + ".csv")
    # 動画の読み込み
    video_capture = cv2.VideoCapture(video_directory + "/" + file)

    # フレームカウントの初期化
    frame_count = 0

    # 人差指の指先の座標履歴を保持するための変数
    history_length = 16
    point_history = deque(maxlen=history_length)

    while video_capture.isOpened():
        if frame_count >= args.frames:
        # 指定フレームより超えると終了
            break

        # カメラ画像取得
        ret, frame = video_capture.read()
        if ret is False:
            break
        frame_width, frame_height = frame.shape[1], frame.shape[0]

        # MediaPipeで扱う画像は、OpenCVのBGRの並びではなくRGBのため変換
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 画像をリードオンリーにしてposes検出処理実施
        rgb_image.flags.writeable = False
        pose_results = pose.process(rgb_image)
        rgb_image.flags.writeable = True

        # 有効なランドマークが検出された場合、ランドマークを描画
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks,
                                        mp_pose.POSE_CONNECTIONS)

            # ランドマーク座標の計算
            landmark_list = calc_landmark_list(rgb_image, pose_results.pose_landmarks)
            # 人差指の指先座標を履歴に追加
            # point_history.append(landmark_list[ID_FINGER_TIP])

            result = {}
            data_source = {}
            #for i,landmark in enumerate(pose_results.pose_landmarks.landmark):
            
            # ランドマークの取得
            for i,landmark in enumerate(landmark_list):
                result[mp_pose.PoseLandmark(i).name] = {"x":landmark["x"],"y":landmark["y"]}
            data_source["pose"] = result
            data_source["timestamp"] = datetime.now()
            # csvに書き込み
            csv_save.save_data(data_source)

            if frip:
                # ランドマークの取得
                for i,landmark in enumerate(landmark_list):
                    result[mp_pose.PoseLandmark(i).name] = {"x":-landmark["x"],"y":landmark["y"]}
                data_source["pose"] = result
                data_source["timestamp"] = datetime.now()
                # csvに書き込み
                csv_save.save_data(data_source)


        # ディスプレイ表示
        frame = draw_point_history(frame, point_history)
        cv2.imshow('preview', frame)

        # キー入力(ESC:プログラム終了)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        frame_count += 1


# リソースの解放
video_capture.release()
pose.close()
cv2.destroyAllWindows()