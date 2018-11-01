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
https://www.kaggle.com/c/house-prices-advanced-regression-techniques#description
header
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names
"""

# coding: utf-8
import sys
import os
import numpy as np
import pandas as pd
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

labels = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
def make_housing_csv():
  dir_path = os.path.dirname(os.path.abspath(__file__))
  csv_path = dir_path + '/housing/housing.csv'
  data_path = dir_path + '/housing/housing.data'
  if os.path.exists(csv_path):
    print('housing.csv is exists: %s' % csv_path)
    return csv_path

  df = pd.read_csv(data_path, header=None, sep='\s+')
  df.columns = labels
  df.to_csv(csv_path, index=False)

  if os.path.exists(csv_path):
    print('housing.csv is created: %s' % csv_path)
    return csv_path  
  # df = df.append([["hello", 1], ["hello world", 2]], ignore_index=True)
  # df.columns = ['review', 'sentiment']
  # np.random.seed(0)
  # df = df.reindex(np.random.permutation(df.index))
  # print(csv_path)
  # df.to_csv(csv_path, index=False)
  # return csv_path
  # df.to_csv(output_file_path, index=False)
  # df = pd.DataFrame()
  # for s in ('test', 'train'):
  #   for l in ('pos', 'neg'):
  #     path = '/aclImdb/%s/%s' % (s, l)
  #     path2 = dir_path + path
  #     # ファイル名だけ取り出す
  #     for file in os.listdir(path2):
  #       # ファイル名とpathを結合
  #       with open(os.path.join(path2, file), "r", encoding="utf-8") as f:
  #         text = f.read()
  #         df = df.append([[text, labels[l]]], ignore_index=True)

  # df.columns = ['review', 'sentiment']
  # np.random.seed(0)
  # df = df.reindex(np.random.permutation(df.index))
  # df.to_csv(csv_path, index=False)
  # ここで結果判定かな。。

def load_data(data_type='train'):
  file_path = os.path.dirname(os.path.abspath(__file__)) + '/house_prices/train.csv'
  logger.debug('load_data : ' + file_path)
  return file_path

def load_test_data(data_type='train'):
  file_path = os.path.dirname(os.path.abspath(__file__)) + '/house_prices/test.csv'
  logger.debug('load_test_data : ' + file_path)
  return file_path

# configに変更すること
def output_dir_path():
  parenr_dir_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../')
  logger.debug(' parenr_dir_path : ' + parenr_dir_path)
  output_dir_path = parenr_dir_path + '/reports/figures/house_prices/'
  return output_dir_path
