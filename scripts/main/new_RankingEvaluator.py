import nltk
import random
from nltk.corpus import movie_reviews, subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.classify import NaiveBayesClassifier
from new_FeatureExtractor import Sentence
from nltk.sentiment.util import mark_negation, extract_unigram_feats


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
        # corpus: [({'source':"", 'question':"", 'answer':""}, <LABEL>)]
        # model: [(Sentence({...}), <LABEL>)]
        # tests: [([...], <LABEL>)]
        # words: [...]
        self.model = [(Sentence(sample), label) for (sample, label) in corpus]
        self.tests = [(sntc.get_all_words(), label)
                      for (sntc, label) in self.model]
        self.sentim_analyzer = SentimentAnalyzer()
        self.words = []

        for (sentence, label) in self.model:
            self.words.extend(sentence.get_all_words())

        self.freq_words = nltk.FreqDist(self.words)

        self.word_features = list(self.freq_words.keys())[:3000]
        self.featuresets = [(sent.extract_word_features(
            self.word_features), label) for (sent, label) in self.model]

        # [({'source':"", 'question':"", 'answer':""}, <LABEL>)]
        all_words_neg = self.sentim_analyzer.all_words(
            [mark_negation(doc) for doc in self.tests])

        unigram_feats = self.sentim_analyzer.unigram_word_feats(
            all_words_neg, min_freq=4)

        self.sentim_analyzer.add_feat_extractor(
            extract_unigram_feats, unigrams=unigram_feats)

        training_set = self.sentim_analyzer.apply_features(self.tests)

        self.trainer = NaiveBayesClassifier.train
        self.classifier = self.sentim_analyzer.train(
            self.trainer, training_set)

    def predict(self, instance):
        instance = Sentence(instance)
        print(self.sentim_analyzer.classify(instance.get_all_words()))

    def test(self, tests):
        sentences = [(Sentence(test[0]), test[1]) for test in tests]
        test = [(sent.get_all_words(), label) for (sent, label) in sentences]
        test_set = self.sentim_analyzer.apply_features(test)
        print(self.sentim_analyzer.evaluate(test_set))


if __name__ == '__main__':
    f = open('input.txt')

    corpus = [eval(line) for line in f.readlines()]

    f.close()
    test = {'question': "Who are interns in enocta?",
            'source': "Adnan and Asrin are interns in Enocta Technologies in Ankara",
            'answer': "Adnan and Asrin"}

    model = RankingEvaluator()
    model.fit(corpus)
    model.predict(test)
