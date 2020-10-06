import nltk
import random
import sklearn
from nltk.corpus import movie_reviews, subjectivity
from nltk.classify import NaiveBayesClassifier, svm
from new_FeatureExtractor import Sentence
from nltk.sentiment.util import mark_negation, extract_unigram_feats
from sklearn.svm import LinearSVC


class RankingEvaluator:
    # [({'source':"", 'question':"", 'answer':""}, <LABEL>)]
    def __init__(self):
        self.model = []
        self.tests = []
        self.words = []
        self.freq_words = None
        self.word_features = []
        self.featuresets = []

    def fit(self, corpus):
        # corpus: [{'question':'str','answer':'str','source':'str'}, 'label']

        self.classifier = nltk.classify.SklearnClassifier(LinearSVC())
        self.dataset = [(Sentence(data).extract_morphological_features(),label) for (data,label) in corpus]
        self.classifier.train(self.dataset)

    def predict(self, instance):
        instance = Sentence(instance)
        print(self.classifier.classify(instance.extract_morphological_features()))

    def test(self, tests):
        testset = [(Sentence(test_data).extract_morphological_features(),label) for (test_data,label) in tests]
        predict = [self.classifier.classify(sentence) == label for (sentence,label) in testset]
        print("Accuracy: {0}".format(sum(predict)/len(predict)))


if __name__ == '__main__':
    positive_samples = open('tests/positive_samples.txt')
    negative_samples = open('tests/negative_samples.txt')

    corpus = []
    corpus.extend([(eval(line), True)
                   for line in positive_samples.readlines()])
    corpus.extend([(eval(line), False)
                   # line {'question':'str','answer':'str','source':'str'}
                   for line in negative_samples.readlines()])

    random.shuffle(corpus)
    train_set = corpus[:int(len(corpus)*.7)]
    tests_set = corpus[int(len(corpus)*.7):]
    
    positive_samples.close()
    negative_samples.close()
    
    re = RankingEvaluator()
    re.fit(train_set)
    re.test(tests_set)
