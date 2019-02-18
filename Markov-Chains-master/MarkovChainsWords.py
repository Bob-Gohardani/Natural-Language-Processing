import random

txt = []
with open('got.txt', 'r') as file:
    txt = file.read().split(' ')

ngrams = dict()

for i in range(len(txt)-1):
    nextWord = txt[i]

    if nextWord not in ngrams:
        ngrams[nextWord] = []
    ngrams[nextWord].append(txt[i+1])


def markovIt():
    currentWord = random.choice(txt)
    result = currentWord
    for i in range(0, 20):
        if not currentWord in ngrams:
            break
        possibilities = ngrams[currentWord]
        nextElement = random.choice(possibilities)
        result += ' ' + nextElement
        currentWord = result.rsplit(' ', 1)[1]


    print(result)

markovIt()

