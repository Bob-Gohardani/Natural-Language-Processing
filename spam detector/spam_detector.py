
# MultinomialNB means that our x data is countable so we use multinomial distribution when calculating P(x|y)
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

# read data with pandas
data = pd.read_csv('spambase.data').as_matrix()

# this line will mix all rows of data so when we select test/train data it is randomized
np.random.shuffle(data)

# all rows, all columns except last one
X = data[:, :48]
Y = data[:, -1]

# all rows except last 100 and all columns
Xtrain = X[:-100,]

Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

# create the model based on multinomial naive bayes
model = MultinomialNB()

 # train the model with Xtrain and Ytrain dataset
model.fit(Xtrain, Ytrain)

# test the model with test data and find the results (this is the cross-validaton section since we know results of test data)
print(model.score(Xtest, Ytest))

from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print(model.score(Xtest, Ytest))

