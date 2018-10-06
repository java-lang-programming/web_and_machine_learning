# coding: utf-8
import pandas as pd
from resource import aclImdb
import sys
sys.path.append('..')


class Trainer():
    def __init__(self):
        self._resource = aclImdb.Resource()

    def train(self):
        df = pd.DataFrame()
        df = pd.read_csv(self._resource.csv_path)
        print(df.head(3))
