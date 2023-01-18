import csv
from datetime import datetime

# test_data = {"pose" : {"hoge": {"x": 1.0, "y": 2.0}, "fuga": {"x": 1.0, "y": 2.0}, "san": {"x": 1.0, "y": 2.0}, "hu": {"x": 1.0, "y": 2.0}}, "emotion": 1, "timestamp": datetime.now()}

class CsvSaveService:

  def __init__(self, filepath: str):
    self.filepath = filepath
    pass

  @staticmethod
  def convert_data(data):
    res_data = dict()
    for bone_name, pos_dict in data["pose"].items():
      for pos_label, pos in pos_dict.items():
        print(f"{bone_name}_{pos_label}") 
        print(pos)
        res_data[f"{bone_name}_{pos_label}"] = pos
    res_data["emotion"] = data["emotion"]
    res_data["timestamp"] = data["timestamp"]
    return res_data

  def save_data(self, data):
    res = self.convert_data(data)
    print(res.keys())
    with open(self.filepath, "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=res.keys())
      writer.writeheader()
      writer.writerow(res)
    
# csv_save = CsvSaveService("hoge.csv")
# data = csv_save.save_data(test_data)
# print(data)
