
設計
https://www.kaggle.com/c/house-prices-advanced-regression-techniques#description
===================================
提出

https://www.kaggle.com/c/house-prices-advanced-regression-techniques#evaluation

==================================
評価

https://mathwords.net/rmsemae




==================================

1回目
1. データをDL OK
2. dfのdataset profileでデータの相関関係をみる。 OK

抽出
OVerRallQual
YearBUilt
TotalBsmtSF
GrLivArea
FUllbath
GarageCars
GarageAreas

# ここは自動化したいな。。。
cols = ['OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'GarageCars', 'GarageArea']

3. 2で相関の高そうな値を5つほど選んで、ヒートマップを出力。具体的な値を確認する。

OverallQual 0.79

3. 最も、相関関係のあるデータを使ってモデルを作成する。 OK

4. 提出 OK
submissionを一度作成して提出してみる。

4205...とか。

4. 切片と傾きを出す。 OK

切片と傾きを出す。
線形回帰直線を引く。全然あかん。これはロバストでもダメかと。

====
次
重回帰モデルでトレーニング


モデルの性能を数値化する。
1. MSE
2. R^2をだす。


5. RMSE(平均平方二乗誤差)を計算する



ランダムフォレスト
deepまで



## フォルダ構成

dataset

機械学習に必要なデータを置く。
必要な分析の単位でフォルダを分ける。

resource

datasetに格納したデータを取得する。

train

データを読み込んでトレーニングする。

## 文法チェック

flake8を利用。

flake8 ./resource/*
flake8 ./train/*

## importの順番

import-orderを利用

// file
import-order --only-file vectorizer.py



License
-------

Copyright 2018 Masaya Suzuki.

Licensed to the Apache Software Foundation (ASF) under one or more contributor
license agreements.  See the NOTICE file distributed with this work for
additional information regarding copyright ownership.  The ASF licenses this
file to you under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License.  You may obtain a copy of
the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
License for the specific language governing permissions and limitations under
the License.
