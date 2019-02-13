from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# we use special encoding since some characters here are not standard
df = pd.read_csv("spam.csv", encoding="ISO-8859-1")

# these columns are empty so we just drop them
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

# rename the columns from v1 and v2
df.columns = ['labels', 'data']

# create binary labels form ham/smap
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['b_labels'].values

# here we create an object of count vectorizer class
count_vectorizer = CountVectorizer(decode_error="ignore")

# here we get string data, count each word and find its probablity based on repetition and whole count of words
X = count_vectorizer.fit_transform(df['data'])

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)

model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print(model.score(Xtrain, Ytrain))
print(model.score(Xtest, Ytest))


# visualize the data to see most frequent words in spam and ham:
def visualize(label):
  words = ''
  for msg in df[df['labels'] == label]['data']:
    msg = msg.lower()
    words += msg + ' '
  wordcloud = WordCloud(width=600, height=400).generate(words)
  plt.imshow(wordcloud)
  plt.axis('off')
  plt.show()


visualize('spam')
visualize('ham')


df['predictions'] = model.predict(X)

# where we have different predictions and actual labels get the text
sneaky_spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
for msg in sneaky_spam:
  print(msg)

# where we have different predictions and actual labels get the text
not_actually_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
for msg in not_actually_spam:
  print(msg)