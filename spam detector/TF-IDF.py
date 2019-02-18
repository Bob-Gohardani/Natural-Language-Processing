docA = "the cat sat on my face"
docB = "the dog sat on my bed"

#bag of words
bowA = docA.split(" ")
bowB = docB.split(" ")

wordSet = set(bowA).union(set(bowB))

#def fromkeys(iterable, value) : Create a new dictionary with keys from iterable and values set to value.
wordDictA = dict.fromkeys(wordSet, 0)
wordDictB = dict.fromkeys(wordSet, 0)

for word in bowA:
    wordDictA[word] += 1

for word in bowB:
    wordDictB[word] += 1

import pandas as pd
print (pd.DataFrame([wordDictA, wordDictB]))

## my - on - the  : these are repeated in both corpus and dont have any special meanings

## TF-IDF   TF(w) * IDF(w)     
#  tf(w) = (Number of times the word appears in a document) / (Total number of words in the document)
#  idf(w) = log(Number of documents / Number of documents that contain word w )


def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bowCount)
    return tfDict


tfbowA = computeTF(wordDictA, bowA)
tfbowB = computeTF(wordDictB, bowB)

def computeIDF(docList):
    import math
    idfDict = {}
    N = len(docList)
    
    #counts the number of documents that contain a word w
    idfDict = dict.fromkeys(docList[0].keys(),0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] +=1
                
    #divide N by denominator above, take the log of that
    for word, val in idfDict.items():
        idfDict[word]= math.log(N / float(val)) 

    return idfDict

idfs = computeIDF([wordDictA, wordDictB])

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val * idfs[word]
    return tfidf

tfidfbowA = computeTFIDF(tfbowA, idfs)
tfidfbowB = computeTFIDF(tfbowB, idfs)

print (pd.DataFrame([tfidfbowA, tfidfbowB]))



