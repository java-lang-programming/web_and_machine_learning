import numpy as np
import sklearn
import pandas as pd
import pandas_profiling as pdp
from dataset import housing
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

class Housing():

    def __init__(self, means=(), stds=()):
        self._csv_path = housing.load_data()

    def output_correlation_matrix_heat_map(self):
        df = pd.read_csv(self._csv_path)
        logger.debug('head data')
        logger.debug(df.head())

    # レポートの出力場所は設定ファイルで変更にするべき。
    # pandas_profilingでプロファイルレポートを出力する
    def output_profiling_report(self):
        df = pd.read_csv(self._csv_path, parse_dates=True, encoding='UTF-8')
        profile = pdp.ProfileReport(df)
        file_path = housing.output_dir_path() + 'housing_outputfile.html'
        logger.debug('output_profiling_report : ' + file_path)
        profile.to_file(outputfile=file_path)

