# coding: utf-8
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np
from model import bow
# from model import bow
import sys
sys.path.append('..')

class Trainer():
    def __init__(self):
        self._bow = bow.Bow()

    # 英語用
    # ここはdata prossessorクラスをつくべきかも。。。
    # ４０分くらいかかる
    # 計算量https://teratail.com/questions/127509
    def train(self):
        df = pd.DataFrame()
        df = pd.read_csv(self._bow.csv_path)
        train = df.loc[:25000, 'review'].values
        label = df.loc[:25000, 'sentiment'].values
        test_train = df.loc[25000:, 'review'].values
        test_label = df.loc[25000:, 'sentiment'].values

        # print(bow.Bow.tokenizer('a dsd sasa ddsds'));
        # print(bow.Bow.tokenizer_porter('I running'))
        stop = self._bow.stop_words
        #print(stop)
        # print(bow.Bow.tokenizeras('a dsd sasa ddsds'));

        tfidf = TfidfVectorizer(strip_accents=None,
                                lowercase=False,
                                preprocessor=None)



        param_grid = [{'vect__ngram_range': [(1, 1)],
                       'vect__stop_words': [stop, None],
                       'vect__tokenizer': [bow.Bow.tokenizer, bow.Bow.tokenizer_porter],
                       'clf__penalty': ['l1', 'l2'],
                       'clf__C': [1.0, 10.0, 100.0]},
                      {'vect__ngram_range': [(1, 1)],
                       'vect__stop_words': [stop, None],
                       'vect__tokenizer': [bow.Bow.tokenizer, bow.Bow.tokenizer_porter],
                       'vect__use_idf':[False],
                       'vect__norm':[None],
                       'clf__penalty': ['l1', 'l2'],
                       'clf__C': [1.0, 10.0, 100.0]},
                      ]

        lr_tfidf = Pipeline([('vect', tfidf),
                             ('clf', LogisticRegression(random_state=0))])

        gridSearchCV = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)

        gridSearchCV.fit(train, label)
        print('params %s' % gridSearchCV.best_params_)
        print('CV Acuracy: %.3f ' % gridSearchCV.best_score_)
        # print(bow.Bow.preprocessor('</a>This is :) is :( a test :-)!'))

    def train2(self):
        df = pd.DataFrame()
        df = pd.read_csv(self._bow.csv_path)
        train = df.loc[:25000, 'review'].values
        label = df.loc[:25000, 'sentiment'].values
        test_train = df.loc[25000:, 'review'].values
        test_label = df.loc[25000:, 'sentiment'].values
        classes = np.array([0, 1])

        #tokenized = self._bow.tokenizer_without_stop_word('I hava a pen')

        x_train, y_label = self._bow.get_minibatch(self._bow.stream_docs(), size=10)
        #print(x_train)
        #print(y_label)
        vect = HashingVectorizer(decode_error='ignore',
                                 n_features=2**21,
                                 preprocessor=None,
                                 tokenizer=self._bow.tokenizer_without_stop_word)

        clf = SGDClassifier(loss='log', random_state=1, n_iter=1)

        for _ in range(45):
            x_train, train_label = self._bow.get_minibatch(self._bow.stream_docs(), size=1000)
            if not x_train:
                break

            x_train = vect.transform(x_train)
            clf.partial_fit(x_train, train_label, classes=classes)

        x_test_train, test_label = self._bow.get_minibatch(self._bow.stream_docs(), size=5000)
        x_test_train = vect.transform(x_test_train)
        print('accuracy %.3f' % clf.score(x_test_train, test_label))
            







