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
from data_processor import housing as data_processor_housing
from dataset import housing
from model import linear_regression_gd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

class Trainer():
    def __init__(self):
        self._csv_file = housing.load_data()

    def train(self):
        df = pd.read_csv(self._csv_file)
        #print(df[['RM']].values.shape)
        X = df[['RM']].values
        y = df['MEDV'].values
        y = y[:, np.newaxis]
        #print(X)
        #print(y)
        sc_x = StandardScaler()
        sc_y = StandardScaler()
        X_std = sc_x.fit_transform(X)
        y_std = sc_y.fit_transform(y)
        #print(y_std)
        # y_std = sc_y.fit_transform(y)
        instance = linear_regression_gd.LinearRegressionGD()
        #print(X_std)
        #print(y_std)
        instance.fit(X_std, y_std)
        # plt.figure()
        # plt.plot(range(1, instance._n_iter+1), instance._cost)
        # plt.ylabel('SSE')
        # plt.xlabel('Epoch')
        # # plt.tight_layout()
        # # plt.savefig('./figures/cost.png', dpi=300)
        # plt.show()
        # plt.savefig(data_processor_housing.file_path.output_training_report)
        # plt.close('all')
        # train = df.loc[:25000, 'review'].values
        # label = df.loc[:25000, 'sentiment'].values
        # test_train = df.loc[25000:, 'review'].values
        # test_label = df.loc[25000:, 'sentiment'].values
        

        #tokenized = self._bow.tokenizer_without_stop_word('I hava a pen')

        # x_train, y_label = self._bow.get_minibatch(self._bow.stream_docs(), size=2)

        return ""

    def train2(self):
        # データ取得は共通か
        #print(X)
        #print(y)
        data_processor = data_processor_housing.Housing()
        result = data_processor.get_liner_data()
        slr = LinearRegression()
        slr.fit(result[0], result[1])
        print('Slope : %.3f' % slr.coef_[0])
        print('Slope : %.3f' % slr.coef_[0])
        plt.figure()
        plt.scatter(result[0], result[1], c='blue')
        plt.plot(result[0], slr.predict(result[0]), c='red')
        plt.ylabel('Average number of rooms [RM]')
        plt.xlabel('price MEDV')
        plt.show()
        plt.savefig(data_processor.output_training_report())
        plt.close('all')









