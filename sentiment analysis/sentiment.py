import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

wordnet_lemmatizer = WordNetLemmatizer()
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

# get positive and negtive reviews from the source file:
positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.find_all('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read())
negative_reviews = negative_reviews.find_all('review_text')

# shuffle positive reviews and select same number of them as negative reviews so our results are balanced
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

# tokenzie a review (given string as input and returns list of tokens)
def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t)>2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]

    return tokens

# this dict maps each token/word to an index. basically each word in here is a feature of our logistic regression model
word_index_map = {}
current_index = 0

positive_tokenized = []
negative_tokenized = []

for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        # there wouldn't be any repeated word in the word_index_map
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

# bag of words tokens to vector
def tokens_to_vectors(tokens, label):
    x = np.zeros(len(word_index_map)+1)
    for t in tokens:
        i = word_index_map[t] # get the index
        x[i] += 1 # update how many times a word has been repeated in the review
    x = x / x.sum() # normalize each vector based how many tokens are inside it
    x[-1] = label
    return x

N = len(positive_reviews) + len(negative_reviews)

# create data matrix with N rows (number of examples) and column number of word_index_map plus the label (positive or negative)
data = np.zeros((N, len(word_index_map)+1))
i = 0
# each tokens inlcudes a review string
for tokens in positive_tokenized:
    xy = tokens_to_vectors(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vectors(tokens, 0)
    data[i,:] = xy
    i += 1

np.random.shuffle(data)

X = data[:, :-1]
Y = data[:, -1]

# last 100 rows are the test (validation) 
Xtrain = X[:-100,]
Ytrain = Y[:-100,]

Xtest = X[-100:,]
Ytest = Y[-100:,]

model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print(model.score(Xtest, Ytest))

# check which words have the max impact
threshold = 0.5
for word, index in word_index_map.items():
    weight = model.coef_[0][index]
    if weight > threshold or weight < -1 * threshold:
        print(word, weight)