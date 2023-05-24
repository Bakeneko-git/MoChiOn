import os
import glob

# ファイルが存在するディレクトリを指定します。
path = './*.csv'

# このディレクトリ内のすべてのファイルを取得します。
files = glob.glob(path)

print(files)

for i, file in enumerate(sorted(files), 39):
    # ファイルの拡張子を取得します。
    extension = os.path.splitext(file)[1]
    
    # 新しいファイル名を生成します。
    new_name = 'swing_{:03d}_frip{}'.format(i, extension)
    
    # ファイルをリネームします。
    os.rename(file, new_name)
