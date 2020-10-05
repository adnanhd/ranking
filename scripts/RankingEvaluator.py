import nltk
import random
from nltk.corpus import movie_reviews, subjectivity
from FeatureExtractor import Sentence


class RankingModel:
    # [{'source':"", 'question':"", 'answer':""}]
    def __init__(self, documents=None):
        self.corpus = documents
        self.sentences = [Sentence(document) for document in documents]
        self.words = []

    def train(X, Y):
        pass

    def fit(X):
        pass

        for sentence in self.sentences:
            self.words.extend(sentence.get_words())

    def get_freq_dist(self):
        return nltk.FreqDist(self.words)


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]
# word_features[0:10]


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)  # will return either True or False

    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]

if __name__ == '__main__':
    inputs = {'question': "Who are interns in enocta?",
              'source': "Adnan and Asrin are interns in Enocta Technologies in Ankara",
              'answer': "Adnan and Asrin"}

    q = Sentence(inputs)
