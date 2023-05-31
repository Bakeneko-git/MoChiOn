import cv2

class Camera:
    
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def capture(self):
      """カメラデータを取得する

      Yields:
          _type_: カメラから取得した配列データ(BGR)
      """
      while self.cap.isOpened():
          ret, frame = self.cap.read()
          if ret:
              yield frame
          else:
              break
  
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()