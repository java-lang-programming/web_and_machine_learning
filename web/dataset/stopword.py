# coding: utf-8
import sys
sys.path.append('..')
import os
import numpy
from nltk.corpus import stopwords

# nltk.download('stopwords')
def stop_words(lang='english'):
  return stopwords.words(lang)
