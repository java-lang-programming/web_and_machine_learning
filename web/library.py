import numpy as np
import sklearn
import pandas as pd
import nltk
#from dataset import titanic, aclImdb
# from train import housing as train_housing
#from train import house_prices as train_house_prices
from train import aclImdb as train_aclImdb
from dataset import stopword
from resource import aclImdb as resource_aclImdb
# from data_processor import housing
# from data_processor import house_prices
from model import aclImdb
from model import linear_regression_gd


#nltk.download('stopwords')
print(np.__version__)
print(sklearn.__version__)

# instance = train_housing.Trainer()
# #print(instance.output_correlation_matrix_heat_map())

# print(instance.validate_train_robust())

# 一連の流れ メモる

#データ
# instance = train_house_prices.Trainer()
#instance.output_profiling_report()
#instance.output_correlation_matrix_heat_map()
# instance.residual_plot_jyukaiki()
#csv = titanic.load_data()
#print(csv)

#train = pd.read_csv(csv)
#print(train)

# trainer = train_aclImdb.Trainer()
# print(trainer.train2())

resource_aclImdb = resource_aclImdb.Resource()
resource_aclImdb.create_classifier()

# model = aclImdb.Model()
# trainer = train_aclImdb.Trainer()
# model.save_classifier(trainer.train2(), 'classifier')

# trainer = train_aclImdb.Trainer()
# trainer.save(trainer.train2(), 'classifier')
