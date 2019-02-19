import bs4 as bs
import urllib.request
import re
import nltk
import heapq

# downloading text from wikipedia
scraped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/God_of_War_(2018_video_game)') # download data via beautifulsoup and save it as variable
article = scraped_data.read() # read data with all html tags into another variable
parsed_article = bs.BeautifulSoup(article, 'lxml')

paragraphs = parsed_article.find_all('p') # get all html elements with p tag (paragraphs)
article_text = ""

for p in paragraphs:
    article_text += p.text

# preprocessing the text
article_text = re.sub(r'\[[0-9]*\]', ' ', article_text) # if you see any [with any number inside it] replace it by ' '
article_text = re.sub(r'\s+', ' ', article_text) # if there is space bigger than ' ' replace it with ' '

formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text) # remove all special characters from the text for tekenization
formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

sentence_list = nltk.sent_tokenize(article_text) # divide the paragraphs string into sentences

# Weighted Frequency of Occurrence for each word
stopwords = nltk.corpus.stopwords.words('english')

word_frequencies = {}
for word in nltk.word_tokenize(formatted_article_text):
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

maximum_frequency = max(word_frequencies.values()) # gives back the word with highest frequency (repetition)
for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

# sentence score based on the frequency of its words
sentence_scores = {}
for sent in sentence_list:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < 30: # we ommit sentences that are longer than 30 words because we dont want long summaries
                if sent not in sentence_scores.keys(): # if it is first word of sentence create new key otherwise add weight of this word to exisitng key of sentence
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

# summerize the article based on N top sentences with highest frequency score
summary_sentences = heapq.nlargest(7, sentence_scores, key = sentence_scores.get) # use heapq library and call nlargest function to retrieve the top 7 sentences with highest scores.

summary = " ".join(summary_sentences)
print(summary)
