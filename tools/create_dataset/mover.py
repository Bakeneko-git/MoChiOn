import glob
import shutil
import os


"""README
  このファイルは実行したディレクトリにあるCSVファイルを特定の条件を持って仕分けるプログラムです
  1. 仕分けるファイルのキーワードをkeywordに指定してください
    例えば、keyword = "frip" とすると、fripというキーワードを含むファイルをすべて移動します
  2. 出力先ディレクトリパスをdestに指定してください
    例えば、dest = "./frip" とすると、fripというディレクトリに移動します
    ない場合は作成してください
  3. 実行してください
    destディレクトリにkeyword指定したファイルが移動していたら終了です
"""

# 仕分けるファイルのキーワード
keyword = "frip"

# すべてのCSVファイルを取得します
path = glob.glob("./*.csv")

# デバック用
# print(path)

# 出力先ディレクトリパス ./frip ディレクトリに移動します
dest = "./" + keyword

# for文でCSVファイルを一つずつ処理します
for i in path:
    if keyword in i: 
      # キーワードに一致した時移動させます
        shutil.move(i, os.path.join(dest, os.path.basename(i)))

