from __future__ import print_function, division
from future.utils import iteritems
import nltk
import random
import numpy as np
from builtins import range
from bs4 import BeautifulSoup
# only use positive reviews here
positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.findAll("review_text")

trigrams = {}
for review in positive_reviews:
    s = review.text.lower()
    # make each review string into tokens
    tokens = nltk.tokenize.word_tokenize(s)
    # we have i-1, i , i+1 so last i-1 is len-2
    for i in range(len(tokens) -2):

        # key to dictionary can tuple but not list
        k = (tokens[i], tokens[i+2])
        if k not in trigrams:
            trigrams[k] = []
        trigrams[k].append(tokens[i+1])

trigram_probabilities = {}
# loop through each trigram
for k, words in trigrams.items():
    # if all middle words for a trigrams are less than two then omit that word
    if len(set(words)) > 1:
        d = {}
        n = 0
        # loop through all possible words for a trigram and count each one's repetition
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] += 1
            n += 1
        for w,c in d.items():
            d[w] = float(c)/ n
        # key is the previous and next word : value is each possibility and it's probability
        trigram_probabilities[k] = d


                # d[w] = float(c)/ n
def random_sample(d):
    # find a random number and if add up of all probabilities is smaller than random then select that word
    r = random.random()
    cumulative = 0
    for w, p in iteritems(d):
        cumulative += p
        if r < cumulative:
            return w

def test_spinner():
    review = random.choice(positive_reviews)
    s = review.text.lower()
    print("Original", s)

    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens)-2):

        # for only 1 in 5 words/ tokens do this
        if random.random() < 0.2:
            k = (tokens[i], tokens[i+2])
            if k in trigram_probabilities:
                w = random_sample(trigram_probabilities[k])
                tokens[i+1] = w
    print("Modified: ")
    print(" ".join(tokens).replace(" .", ".").replace(" '", "'").replace(" ,", ",").replace("$ ", "$").replace(" !", "!"))


test_spinner()
test_spinner()

        
