# coding: utf-8
from dataset import aclImdb
from dataset import stopword
import sys
sys.path.append('..')
import re
from nltk.stem.porter import PorterStemmer


class Bow:
    def __init__(self):
        self._csv_path = aclImdb.make_review_csv()
        self._stop_words = stopword.stop_words()

    @property
    def csv_path(self):
        return self._csv_path

    @classmethod
    def preprocessor(self, text):
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        text = re.sub('[\W]+', ' ', text.lower()) +\
            ' '.join(emoticons).replace('-', '')
        return text

    @property
    def stop_words(self):
        return self._stop_words

    @classmethod
    def tokenizer(self, text):
       return text.split()

    @classmethod
    def tokenizer_porter(self, text):
       porter = PorterStemmer()
       return [porter.stem(word) for word in text.split()]
