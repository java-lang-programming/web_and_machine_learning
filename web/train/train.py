# coding: utf-8
import sys
sys.path.append('..')  # 親ディレクトリのファイルをインポートするための設定
import pandas as pd
from dataset import titanic

csv = titanic.load_data()
print(csv)

train = pd.read_csv(csv)
print(train)

make_review_csv