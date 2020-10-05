import nltk
import random
from nltk import pos_tag
from nltk import ne_chunk
from nltk.sentiment.util import *
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, subjectivity


class FeatureExtractor:
    def __init__(self, featurable=''):
        self.features = {}
        self.tokens = word_tokenize(featurable.lower())
        self.pos_tags = pos_tag(self.tokens)
        self.ner_tree = ne_chunk(self.pos_tags)

        self.setFeatureValue('__length__',len(featurable))
        self.setFeatureValue('__wordct__',len(self.tokens))
        self.setFeatureValue('__pos__',None)
        self.setFeatureValue('__neu__',None)
        self.setFeatureValue('__neg__',None)

        porter = PorterStemmer()
        for ((token, pos), ner) in self.ner_tree.pos():
            self.incFeatureValue(pos)
            self.incFeatureValue(porter.stem(token))
            if ner != "S":
                self.incFeatureValue(ner)

    def isFeatureExists(self, key):
        try:
            self.features[key]
            return True
        except KeyError as e:
            return False

    def incFeatureValue(self, key):
        if self.isFeatureExists(key):
            self.features[key] += 1.0
        else:
            self.features[key] = 1.0

    def decFeatureValue(self, key):
        if self.isFeatureExists(key):
            self.features[key] -= 1.0
        else:
            self.features[key] = - 1.0

    def setFeatureValue(self, key, value):
        self.features[key] = value

    def getFeatureValue(self, key):
        if self.isFeatureExists(key):
            return self.features[key]

    def getFeaturesLenght(self):
        return len(self.features)

    def getFeatures(self):
        return self.features


class Question:
    def __init__(self, args=('', '', '')):
        self.question = args[0]
        self.sentence = args[1]
        self.answer = args[2]

        self.questionFeature = FeatureExtractor(word_tokenize(self.question))
        self.sentenceFeature = FeatureExtractor(word_tokenize(self.sentence))

    def getSentenceFeatures(self):
        return self.sentenceFeature

    def getQuestionFeatures(self):
        return self.questionFeature

class RankingModel:
    def __init__(self, document):
        self.corpora = nltk.sent_tokenize(document)

if __name__ != '__main__':
    SenQuesPair = ("Who are interns in enocta?",
                     "Adnan and Asrin are interns in Enocta Technologies in Ankara",
                     "Adnan and Asrin")

    q = Question(SenQuesPair)
    f = FeatureExtractor(word_tokenize("the men's book"))
    
    print(q.getQuestionFeatures().getFeatures())
    print(q.getSentenceFeatures().getFeatures())
    
    print(f.getFeatures())
    

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
        features[w] = (w in words) # will return either True or False

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

n_instances = 100

subj_docs = [(sent, 'subj') for sent in subjectivity.sents(cetegories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(cetegories='obj')[:n_instances]]

train_subj_docs = subj_docs[:80]
test_subj_docs = subj_docs[80:100]
train_obj_docs = obj_docs[:80]
test_obj_docs = obj_docs[80:100]
training_docs = train_subj_docs+train_obj_docs
testing_docs = test_subj_docs+test_obj_docs

sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])

unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
