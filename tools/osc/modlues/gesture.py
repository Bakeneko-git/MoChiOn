from nnabla.utils.nnp_graph import NnpLoader
import numpy as np

class Gesture:
    
    def __init__(self, modelPath: str):
      nnp = NnpLoader(modelPath)
      self.net = nnp.get_network('Runtime', batch_size=1)
      self.x = self.net.inputs['Input']
      self.y = self.net.outputs["y'"]

    def estimate(self, pose_storage: np.array) -> np.ndarray:
      """150フレーム分の姿勢推定データからポーズ推定を行う

      Args:
          pose_storage (np.array): 150, 

      Returns:
          np.ndarray: _description_
      """
      self.x.d = pose_storage
      self.y.forward()
      ans = np.array(self.y.d.copy())
      ans = ans.flatten()
      return ans
