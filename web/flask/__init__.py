import os
import pickle
import numpy as np
import re
from flask import Flask, make_response, jsonify
from sklearn.feature_extraction.text import HashingVectorizer

app = Flask(__name__)

@app.route('/emotion', methods=['GET'])
def hello():
    vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)
    # ファイル設定ファイルから。
    # parenr_dir_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../')
    # parenr_dir_path = parenr_dir_path + '/models/movieclassifier/pkl_objects'
    # stopwords_path = parenr_dir_path + '/stopwords.pkl'
    # stop = pickle.load(open(stopwords_path, 'rb'))
    
    parenr_dir_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../')
    parenr_dir_path = parenr_dir_path + '/models/movieclassifier/pkl_objects'
    file_path = parenr_dir_path + '/classifier.pkl'
    clf = pickle.load(open(file_path, 'rb'))
    label = {0: 'negative', 1: 'positive'}
    example = ['I love this movie.']

    X = vect.transform(example)
    result = { "emotion": label[clf.predict(X)[0]], "parcent": np.max(clf.predict_proba(X)) * 100 }
    return make_response(jsonify(result))

def tokenizer(text):
    # チェック処理を入れる
    parenr_dir_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../')
    parenr_dir_path = parenr_dir_path + '/models/movieclassifier/pkl_objects'
    stopwords_path = parenr_dir_path + '/stopwords.pkl'
    stop = pickle.load(open(stopwords_path, 'rb'))
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + \
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
