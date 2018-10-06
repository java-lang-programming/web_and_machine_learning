import numpy as np
import sklearn
import pandas as pd
#from dataset import titanic, aclImdb
from train import aclImdb

print(np.__version__)
print(sklearn.__version__)

#csv = titanic.load_data()
#print(csv)

#train = pd.read_csv(csv)
#print(train)

trainer = aclImdb.Trainer()
print(trainer.train())

