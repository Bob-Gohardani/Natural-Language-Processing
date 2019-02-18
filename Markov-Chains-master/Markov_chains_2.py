import random

names = []
order = 3
ngrams = dict()
beginning = []

with open('data.txt', 'r') as myfile:
    names = myfile.read().split('\n')

for j in range(len(names)):
    txt = names[j]
    for i in range(0, len(txt) - order):
        gram = txt[i:i + order]
        if i == 0:
            beginning.append(gram)

        if gram not in ngrams:
            ngrams[gram] = []
        ngrams[gram].append(txt[i + order])


def markovIt():
    currentGram = random.choice(beginning)

    result = currentGram
    for i in range(0, 10):
        if not currentGram in ngrams:
            break
        possibilites = ngrams[currentGram]
        nextElement = random.choice(possibilites)
        result += nextElement
        currentGram = result[-1 * order:]
    print(result)

markovIt()






