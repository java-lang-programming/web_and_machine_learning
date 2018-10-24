import numpy as np
import sklearn
import pandas as pd
import nltk
#from dataset import titanic, aclImdb
from train import housing as train_housing
from dataset import stopword
from data_processor import housing
from model import aclImdb
from model import linear_regression_gd


#nltk.download('stopwords')
print(np.__version__)
print(sklearn.__version__)

instance = train_housing.Trainer()
#print(instance.output_correlation_matrix_heat_map())

print(instance.validate_train_robust())

#csv = titanic.load_data()
#print(csv)

#train = pd.read_csv(csv)
#print(train)

# trainer = aclImdb.Trainer()
# print(trainer.train2())

# model = aclImdb.Model()
# trainer = train_aclImdb.Trainer()
# model.save(stopword.stop_words(), 'stopwords', trainer.train2(), 'classifier')

# trainer = train_aclImdb.Trainer()
# trainer.save(trainer.train2(), 'classifier')
