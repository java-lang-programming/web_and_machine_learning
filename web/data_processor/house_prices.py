import numpy as np
import sklearn
import pandas as pd
import pandas_profiling as pdp
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from dataset import house_prices
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

class HousePrices():

    def __init__(self, means=(), stds=()):
        self._csv_path = house_prices.load_data()
        self._csv_test_path = house_prices.load_test_data()

    # ヒートマップの改善をする
    def output_correlation_matrix_heat_map(self):
        df = pd.read_csv(self._csv_path)
        logger.debug('head data')
        # logger.debug(np.array(df.values())
        # print(df.values())
        #logger.debug(df.values())
        # logger.debug(df.columns)
        # logger.debug(df.values)
        cols = ['OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'GarageCars', 'GarageArea', 'SalePrice']
        cm = np.corrcoef(df[cols].values.T)
        file_path = house_prices.output_dir_path() + 'output_heat_map.png'
        logger.debug('output_heat_map : ' + file_path)
        plt.figure()
        sns.set(font_scale=1.5)
        fm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                         annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
        plt.savefig(file_path)
        plt.close('all')


    # レポートの出力場所は設定ファイルで変更にするべき。
    # pandas_profilingでプロファイルレポートを出力する
    # folderがない場合は、フォルダを作成でき流ようにする
    def output_profiling_report(self):
        df = pd.read_csv(self._csv_path, parse_dates=True, encoding='UTF-8')
        profile = pdp.ProfileReport(df)
        file_path = house_prices.output_dir_path() + 'house_prices_outputfile.html'
        logger.debug('output_profiling_report : ' + file_path)
        profile.to_file(outputfile=file_path)

    def output_training_report(self):
        file_path = house_prices.output_dir_path() + 'house_prices_training.png'
        logger.debug('output_training_report : ' + file_path)
        return file_path

    def output_submission_report(self, result):
        df = pd.read_csv(self._csv_test_path)
        df['SalePrice'] = result
        file_path = house_prices.output_dir_path() + 'house_prices_submission.csv'
        df.to_csv(file_path, columns=['Id', 'SalePrice'], index=False)

    def get_liner_data(self):
        df = pd.read_csv(self._csv_path)
        X = df[['RM']].values
        y = df['MEDV'].values
        y = y[:, np.newaxis]
        sc_x = StandardScaler()
        sc_y = StandardScaler()
        X_std = sc_x.fit_transform(X)
        y_std = sc_y.fit_transform(y)
        return (X_std, y_std)

    def get_normal_data(self):
        df = pd.read_csv(self._csv_path)
        X = df[['OverallQual']].values
        y = df['SalePrice'].values
        y = y[:, np.newaxis]
        return (X, y)

    def get_normal_all_data(self):
        df = pd.read_csv(self._csv_path)
        #X = df.iloc[:, :-1].values
        # 'C1', 'C2'  Alley
        cols = ['1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtFinSF1']
        X = df[cols].values
        y = df['SalePrice'].values
        return (X, y)

    def get_test_data(self):
        df = pd.read_csv(self._csv_test_path)
        X = df[['OverallQual']].values
        #y = df['SalePrice'].values
        #y = y[:, np.newaxis]
        return X

    def get_cross_data(self):
        df = pd.read_csv(self._csv_path)
        X = df.iloc[:, :-1].values
        y = df['MEDV'].values
        #y = y[:, np.newaxis]
        return (X, y)
