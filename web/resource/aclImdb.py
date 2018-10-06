# coding: utf-8
import sys
sys.path.append('..')
from dataset import aclImdb

class Resource():
  def __init__(self):
    self._csv_path = aclImdb.make_review_csv()

  @property
  def csv_path(self):
    return self._csv_path
