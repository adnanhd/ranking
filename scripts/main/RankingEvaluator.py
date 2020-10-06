import nltk
import random
from nltk.corpus import movie_reviews, subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.classify import NaiveBayesClassifier
from FeatureExtractor import Sentence
from nltk.sentiment.util import mark_negation, extract_unigram_feats

sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words(
    [mark_negation(doc) for doc in training_docs])

unigram_feats = sentim_analyzer.unigram_word_feats(
    all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(
    extract_unigram_feats, unigrams=unigram_feats)

training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)


class RankingModel:
    # [({'source':"", 'question':"", 'answer':""}, 'label')]
    def __init__(self):
        self.model = {} # {Sentence():'label'}
        self.words = []
        self.sentim = SentimentAnalyzer()
        self.classify = NaiveBayesClassifier()


    '''
     @param train = [({'source':"", 'question':"", 'answer':""}, bool)]
    '''
    def fit(self, train):
        self.corpus = train

        for document in documents:
            self.model[Sentence(document[0])] = document[1]

        for sentence in self.sentences:
            self.words.extend(sentence.get_words())

    def predict(self, test): # {'source':"", 'question':"", 'answer':""}
        sent = Sentence(test)
        return sent.get_feature_vector(self.words)

        for word in list(self.get_freq_dist().keys())[:3000]:
            test_features[word] = (w in self.words)  # will return either True or False

        return features

    def get_freq_dist(self):
        return nltk.FreqDist(self.words)

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)  # will return either True or False

    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]


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






if __name__ == '__main__':
    inputs = {'question': "Who are interns in enocta?",
              'source': "Adnan and Asrin are interns in Enocta Technologies in Ankara",
              'answer': "Adnan and Asrin"}

    q = Sentence(inputs)

    n_instances = 100

    subj_docs = [(sent, 'subj')
                 for sent in subjectivity.sents(categories='subj')[:n_instances]]
    obj_docs = [(sent, 'obj')
                for sent in subjectivity.sents(categories='obj')[:n_instances]]

    train_subj_docs = subj_docs[:80]
    test_subj_docs = subj_docs[80:100]
    train_obj_docs = obj_docs[:80]
    test_obj_docs = obj_docs[80:100]
    training_docs = train_subj_docs+train_obj_docs
    testing_docs = test_subj_docs+test_obj_docs

    sentim_analyzer = SentimentAnalyzer()
    all_words_neg = sentim_analyzer.all_words(
        [mark_negation(doc) for doc in training_docs])

    unigram_feats = sentim_analyzer.unigram_word_feats(
        all_words_neg, min_freq=4)
    sentim_analyzer.add_feat_extractor(
        extract_unigram_feats, unigrams=unigram_feats)

    training_set = sentim_analyzer.apply_features(training_docs)
    test_set = sentim_analyzer.apply_features(testing_docs)

    trainer = NaiveBayesClassifier.train
    classifier = sentim_analyzer.train(trainer, training_set)
