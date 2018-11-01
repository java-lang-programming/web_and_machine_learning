import numpy as np
import pickle
import re
import os
from sklearn.feature_extraction.text import HashingVectorizer


parenr_dir_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/models/movieclassifier/pkl_objects')

print(parenr_dir_path)

stopwords_path = parenr_dir_path + '/stopwords.pkl'

stop = pickle.load(open(stopwords_path, 'rb'))


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + \
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)


file_path = parenr_dir_path + '/classifier.pkl'

clf = pickle.load(open(file_path, 'rb'))

label = {0: 'negative', 1: 'positive'}
example = ['this movie is fun.']

X = vect.transform(example)

print(clf.predict(X))
print(clf.predict_proba(X))


print('Prediction: %s\nProbability: %.2f%%' %
      (label[clf.predict(X)[0]],
       np.max(clf.predict_proba(X)) * 100))

