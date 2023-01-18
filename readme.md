### データ構造
```python
class pose_input(TypedDict):
 bonename: str
 positionX: float
 positionY: float

class emotion_input(TypeDict):
 emotion: int

```

### ファイル説明
**main.py**
- エントリポイント

**csv_out.py**
- データを保存する

**pose_input.py**
- ポーズを推定する
- get_pose() : フレームを受け取りデータを返す

**emotion_input.py**
- 感情ラベルを作成する
- get_emotion() : 入力を受け取りデータを返す

