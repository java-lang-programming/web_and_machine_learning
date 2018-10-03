# coding: utf-8
import sys
sys.path.append('..')
import os
import numpy

def load_data(file_name='train.csv'):
  file_path = os.path.dirname(os.path.abspath(__file__)) + '/titanic/' + file_name
  print(file_path) 

  if not os.path.exists(file_path):
    print('No file: %s' % file_name)
    return None 

  return file_path

