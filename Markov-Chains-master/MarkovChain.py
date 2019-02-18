import random

txt = "The unicorn is a legendary creature that has been described since antiquity as a beast with a single large, pointed, spiraling horn projecting from its forehead. The unicorn was depicted in ancient seals of the Indus Valley Civilization and was mentioned by the ancient Greeks in accounts of natural history by various writers, including Ctesias, Strabo, Pliny the Younger, and Aelian.[1] The Bible also describes an animal, the re'em, which some versions translate as unicorn.[1] In European folklore, the unicorn is often depicted as a white horse-like or goat-like animal with a long horn and cloven hooves (sometimes a goat's beard). In the Middle Ages and Renaissance, it was commonly described as an extremely wild woodland creature, a symbol of purity and grace, which could be captured only by a virgin. In the encyclopedias, its horn was said to have the power to render poisoned water potable and to heal sickness. In medieval and Renaissance times, the tusk of the narwhal was sometimes sold as unicorn horn ."
order = 6 # tri-gram

ngrams = dict()
for i in range(0, len(txt)-order):
    gram = txt[i:i+order]

    if gram not in ngrams:
        ngrams[gram] = []
    ngrams[gram].append(txt[i+order])
	

def markovIt():
    currentGram = txt[0:order]
    result = currentGram
    for i in range(0, 100):
        if not currentGram in ngrams:
            break
        possibilites = ngrams[currentGram]
        nextElement = random.choice(possibilites)
        result += nextElement
        currentGram = result[-1 * order:]
        # problem is when we find last 5 characters of word they might be same as last word in the string and there is nothing after that.
    print(result)

markovIt()




