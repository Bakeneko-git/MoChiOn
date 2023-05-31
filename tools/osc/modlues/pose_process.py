from modlues import gesture, image_process

class Pose_process:
  def __init__(self, modelPath: str):
    self.image_process = image_process.Image_process()
    self.gesture = gesture.Gesture(modelPath)
  
  def start(self):
    for frame in self.image_process.start():
      if frame is None:
        yield None
      ans = self.gesture.estimate(frame)
      yield ans