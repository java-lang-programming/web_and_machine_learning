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

    def tokenizer_without_stop_word(self, text):
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        text = re.sub('[\W]+', ' ', text.lower()) +\
            ' '.join(emoticons).replace('-', '')
        tokenized = [w for w in text.split() if w not in self._stop_words]
        return tokenized

    def stream_docs(self):
        with open(self._csv_path, 'r', encoding="utf-8") as csv:
            next(csv)
            for line in csv:
                text, label = line[:-3], int(line[-2])
                yield text, label

    def get_minibatch(self, stream_docs, size):
        docs, y = [], []
        try:
            for _ in range(size):
                text, label = next(stream_docs)
                docs.append(text)
                y.append(label)
        except StopIteration:
            return None, None
        return docs, y

