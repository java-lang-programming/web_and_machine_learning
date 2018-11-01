# coding: utf-8
import pickle
import os
import numpy as np
import pandas as pd
import sys

class Model():

    def save_classifier(self, clf, clf_filename):
        parenr_dir_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../')
        # parent_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
        print(parenr_dir_path)
        model_dir_path = parenr_dir_path + '/models/'
        dest = os.path.join(model_dir_path, 'movieclassifier', 'pkl_objects')
        print(dest)
        if not os.path.exists(dest):
            os.makedirs(dest)

        pickle.dump(clf,
                    open(os.path.join(dest, clf_filename + '.pkl'),'wb'),
                    protocol=4)

    def save_stopwords(self, object, filename, clf, clf_filename):
        parenr_dir_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../')
        # parent_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
        print(parenr_dir_path)
        model_dir_path = parenr_dir_path + '/models/'
        dest = os.path.join(model_dir_path, 'stopwords', 'pkl_objects')
        print(dest)
        if not os.path.exists(dest):
            os.makedirs(dest)

        pickle.dump(object,
                    open(os.path.join(dest, filename + '.pkl'),'wb'),
                    protocol=4)


