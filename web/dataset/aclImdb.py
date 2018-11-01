# Copyright 2018 Masaya Suzuki.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Large Movie Review Dataset
http://ai.stanford.edu/~amaas/data/sentiment/
"""

# coding: utf-8
import sys
sys.path.append('..')
import os
import numpy as np
import pandas as pd

labels = {'pos':1, 'neg':0}
# 間違っているような。。。多分ランダムになってない。
def make_review_csv():
  # ここで作成済みかどうかをチェック
  dir_path = os.path.dirname(os.path.abspath(__file__))
  csv_path = dir_path + '/aclImdb/movie_data.csv'
  if os.path.exists(csv_path):
    print('movie_data.csv is exists: %s' % csv_path)
    return csv_path

  print(dir_path)

  df = pd.DataFrame()
  # df = df.append([["hello", 1], ["hello world", 2]], ignore_index=True)
  # df.columns = ['review', 'sentiment']
  # np.random.seed(0)
  # df = df.reindex(np.random.permutation(df.index))
  # print(csv_path)
  # df.to_csv(csv_path, index=False)
  # return csv_path
  # df.to_csv(output_file_path, index=False)
  # df = pd.DataFrame()
  for s in ('test', 'train'):
    for l in ('pos', 'neg'):
      path = '/aclImdb/%s/%s' % (s, l)
      path2 = dir_path + path
      # ファイル名だけ取り出す
      for file in os.listdir(path2):
        # ファイル名とpathを結合
        with open(os.path.join(path2, file), "r", encoding="utf-8") as f:
          text = f.read()
          df = df.append([[text, labels[l]]], ignore_index=True)

  df.columns = ['review', 'sentiment']
  np.random.seed(0)
  df = df.reindex(np.random.permutation(df.index))
  df.to_csv(csv_path, index=False)
  return csv_path

def load_data(data_type='train'):
  file_path = os.path.dirname(os.path.abspath(__file__)) + '/aclImdb/'
  print(file_path)

  for s in ('test', 'train'):
    for l in ('pos', 'neg'):
      path = '/aclImdb/%s/%s' % (s, l)
      print(path)

  return file_path

