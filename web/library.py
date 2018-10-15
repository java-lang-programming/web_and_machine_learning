import numpy as np
import sklearn
import pandas as pd
import nltk
#from dataset import titanic, aclImdb
from train import aclImdb as train_aclImdb
from dataset import stopword
from model import aclImdb


#nltk.download('stopwords')
print(np.__version__)
print(sklearn.__version__)

#print(stopword.stop_words())

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
