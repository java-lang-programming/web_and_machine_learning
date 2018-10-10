# coding: utf-8
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from model import bow
# from model import bow
import sys
sys.path.append('..')

class Trainer():
    def __init__(self):
        self._bow = bow.Bow()

    # 英語用
    # ここはdata prossessorクラスをつくべきかも。。。
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

        print('sssssssss')

        lr_tfidf = Pipeline([('vect', tfidf),
                             ('clf', LogisticRegression(random_state=0))])

        gridSearchCV = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)

        gridSearchCV.fit(train, label)
        print('params %s' % gridSearchCV.best_params_)
        # print(bow.Bow.preprocessor('</a>This is :) is :( a test :-)!'))



