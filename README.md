
Python Machine Lerning 
===================================

効率よく機械学習を実装しながら、Webを連携させるための機械学習プロジェクトのテンプレート。
発展途上なので、色々と変えていきます。

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
flake8 ./flask/*

## importの順番

import-orderを利用

// file
import-order --only-file vectorizer.py
import-order --only-file vectorizer.py


## 起動
cd flask
python  __init__.py

http://localhost:8000/get

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
