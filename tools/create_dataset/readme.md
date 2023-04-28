# what is this? 
create_datasetモジュールは動画ファイル(mp4)からmediapipeを利用してランドマークをcsvに変換するモジュールです。

# 推奨環境

入力される動画は`.m4`が期待されます。

windows11, python1.10.8にて動作確認済み

**依存環境**

このモジュールは以下のパッケージに依存しています。

- opencv-python 4.7.0.68
- mediapipe 0.9.0.1
- numpy 1.24.1


# 使用方法

動画を変換する手順は以下の通りになります。

1. `main.py` 同レベルに`./video`ディレクトリを作成してください。
2. `./video`ディレクトリ内に`.mp4`の動画ファイルを配置してください
3. `main.py` を実行してください
4. `your_filename.mp4.csv` ファイルが` `main.py`と同レベルに作成されたから完了です。

`main.py` 

動画を読み込んでCSVファイルを作成するファイルです。

`csv_output.py`

csvファイルを読み書きするファイルです。

TODO: mediapipeの実装に依存しているのを解除する。

`landmark.py`

画像を受け取ってmediapipeの処理を行うファイルです。

--- 

`create_dataset.py` : 旧, 動画をcsvに変換するモジュールです。

コマンドライン引数を必ず必要とします。

- `--path` : 動画ファイルのパスです。実行環境からの相対パスもしくは絶対パスが期待されます。
- `--frames` : 動画のフレーム数 デフォルト値として150frameを設定しています。
- `--pat` : deprecated, 現在は使用されていません。無視しても問題ありません。

example: 
```bah
python .\create_dataset.py  --path="video"
```