# coding: utf-8
from dataset import aclImdb
from train import aclImdb as train_aclImdb
from model import aclImdb as model_aclImdb
import sys
sys.path.append('..')


class Resource():
    def __init__(self):
        self._csv_path = aclImdb.make_review_csv()

    @property
    def csv_path(self):
        return self._csv_path

    def create_classifier(self):
        model = model_aclImdb.Model()
        trainer = train_aclImdb.Trainer()
        model.save_classifier(trainer.train2(), 'classifier')
