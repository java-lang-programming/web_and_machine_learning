# coding: utf-8
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from model import bow
from data_processor import housing as data_processor_housing
from dataset import housing
from model import linear_regression_gd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
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
        result = data_processor.get_normal_data()
        slr = LinearRegression()
        slr.fit(result[0], result[1])
        print('Slope : %.3f' % slr.coef_[0])
        print('Slope : %.3f' % slr.intercept_)
        plt.figure()
        plt.scatter(result[0], result[1], c='blue')
        plt.plot(result[0], slr.predict(result[0]), c='red')
        plt.ylabel('Average number of rooms [RM]')
        plt.xlabel('price MEDV')
        plt.show()
        plt.savefig(data_processor.output_training_report())
        plt.close('all')

    def train3(self):
        # データ取得は共通か
        #print(X)
        #print(y)
        data_processor = data_processor_housing.Housing()
        result = data_processor.get_normal_data()
        slr = LinearRegression()
        slr.fit(result[0], result[1])
        print('Slope : %.3f' % slr.coef_[0])
        print('Slope : %.3f' % slr.intercept_)
        plt.figure()
        plt.scatter(result[0], result[1], c='blue')
        plt.plot(result[0], slr.predict(result[0]), c='red')
        plt.ylabel('Average number of rooms [RM]')
        plt.xlabel('price MEDV')
        plt.show()
        plt.savefig(data_processor.output_training_report())
        plt.close('all')

    # ロバスト
    def train_robust(self):
        # データ取得は共通か
        #print(X)
        #print(y)
        data_processor = data_processor_housing.Housing()
        result = data_processor.get_normal_data()
        # residual_metric ：呼び出し可能、オプション
        # メトリックを使用して、多次元目標値y.shape[1] > 1に対して残差の次元数を1に減らします。 デフォルトでは絶対差の合計が使用されます：
        # 絶対値損失
        # https://code-examples.net/ja/docs/scikit_learn/modules/generated/sklearn.linear_model.ransacregressor
        slr = RANSACRegressor(LinearRegression(),
                              max_trials=100,
                              min_samples=50,
                              loss='absolute_loss',
                              residual_threshold=5.0,
                              random_state=0)
        slr.fit(result[0], result[1])
        # 正常値
        inlier_mask = slr.inlier_mask_
        # 外れ値
        outlier_mask = np.logical_not(inlier_mask)
        #print(slr.inlier_mask_)
        #print(np.logical_not(slr.inlier_mask_))
        line_X = np.arange(3, 15, 1)
        line_y = slr.predict(line_X[:, np.newaxis])
        # 縦にする
        #print(line_X[:, np.newaxis])
        print(slr.predict(line_X[:, np.newaxis]))
        print(result[0].shape)
        print(result[0][inlier_mask].shape)
        print(result[0][outlier_mask].shape)
        # 予測値(最小にじょう)
        plt.plot(line_X, line_y, c='red')
        plt.ylabel('Average number of rooms [RM]')
        plt.xlabel('price MEDV')
        plt.show()
        plt.savefig(data_processor.output_training_report())
        plt.close('all')

     
    # ロバスト性能評価
    # 残差プロット出力
    # 平均二乗誤差
    # R^2 
    def validate_train_robust(self):
        data_processor = data_processor_housing.Housing()
        result = data_processor.get_cross_data()
        #print(result[0].shape)
        #print(result[1].shape)
        # トレーニングデータ, テストデータ, ラベルトレーニング, ラベルテスト
        X_train, X_test, y_train, y_test = train_test_split(result[0], result[1], test_size=0.3, random_state=0)
        print(X_train.shape)
        print(X_test.shape)
        slr = LinearRegression()
        slr.fit(result[0], result[1])
        y_train_pred = slr.predict(X_train)
        y_test_pred = slr.predict(X_test)
        print(y_train_pred.shape)
        print(y_test_pred.shape)
        plt.figure()
        plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
        plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='o', label='Test Data')
        plt.xlabel('predicted value')
        plt.ylabel('Residuals')
        plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
        plt.xlim([-10, 50])
        plt.show()
        plt.savefig(data_processor.output_training_report())
        plt.close('all')

