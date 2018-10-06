# coding: utf-8
from dataset import aclImdb
import sys
sys.path.append('..')


class Resource():
    def __init__(self):
        self._csv_path = aclImdb.make_review_csv()

    @property
    def csv_path(self):
        return self._csv_path
