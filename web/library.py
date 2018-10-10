import numpy as np
import sklearn
import pandas as pd
import nltk
#from dataset import titanic, aclImdb
from train import aclImdb
from dataset import stopword


#nltk.download('stopwords')
print(np.__version__)
print(sklearn.__version__)

#print(stopword.stop_words())

#csv = titanic.load_data()
#print(csv)

#train = pd.read_csv(csv)
#print(train)

trainer = aclImdb.Trainer()
print(trainer.train())
