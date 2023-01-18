import csv
from datetime import datetime

class CsvSaveService:

  def __init__(self, filepath: str):
    self.filepath = filepath
    self.is_first = True
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
    with open(self.filepath, "a", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=res.keys())
      if (self.is_first):
        writer.writeheader()
        self.is_first = False
      writer.writerow(res)
    
if __name__ == "__main__": 
  test_data = {"pose" : {"hoge": {"x": 1.0, "y": 2.0}, "fuga": {"x": 1.0, "y": 2.0}, "san": {"x": 1.0, "y": 2.0}, "hu": {"x": 1.0, "y": 2.0}}, "emotion": 1, "timestamp": datetime.now()}
  csv_save = CsvSaveService("hoge.csv")
  data = csv_save.save_data(test_data)
  print(data)
