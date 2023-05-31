from modlues import camera, pose

class Image_process:
  def __init__(self):
    self.camera = camera.Camera()
    self.pose = pose.Pose()
  
  def start(self):
    for frame in self.camera.capture():
      frame = self.pose.estimate_A(frame)
      if frame is None:
        yield None
      yield frame