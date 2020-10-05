import nltk
import random
from nltk import pos_tag
from nltk import ne_chunk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, subjectivity
from nltk.sentiment.util import mark_negation, extract_unigram_feats

class Featurable:
    def __init__(self, featurable=''):
        self.features = {}
        self.tokens = word_tokenize(featurable.lower())
        self.pos_tags = pos_tag(self.tokens)
        self.ner_tree = ne_chunk(self.pos_tags)

        self.set_feature_value('__length__', len(featurable))
        self.set_feature_value('__wordct__', len(self.tokens))
        self.set_feature_value('__pos__', 0)
        self.set_feature_value('__neu__', 0)
        self.set_feature_value('__neg__', 0)

        porter = PorterStemmer()
        for ((token, pos), ner) in self.ner_tree.pos():
            self.inc_feature_value(pos)
            self.inc_feature_value(porter.stem(token))
            if ner != "S":
                self.inc_feature_value(ner)

    def is_feature_exists(self, key):
        try:
            self.features[key]
            return True
        except KeyError:
            return False

    def inc_feature_value(self, key):
        if self.is_feature_exists(key):
            self.features[key] += 1
        else:
            self.features[key] = 1

    def dec_feature_value(self, key):
        if self.is_feature_exists(key):
            self.features[key] -= 1
        else:
            self.features[key] = - 1

    def set_feature_value(self, key, value):
        self.features[key] = value

    def get_feature_value(self, key):
        if self.is_feature_exists(key):
            return self.features[key]

    def get_features_lenght(self):
        return len(self.features)

    def get_features(self):
        return self.features


class Sentence:
    def __init__(self, kwds={'source': "", 'question': "", 'answer': ""}):
        self.question = kwds['question']
        self.source = kwds['source']
        self.answer = kwds['answer']

        self.question_feature = Featurable(self.question)
        self.source_feature = Featurable(self.source)

    def get_question(self):
        return self.question

    def get_source(self):
        return self.source

    def get_answer(self):
        return self.answer

    def get_words(self):
        return word_tokenize(self.question) + word_tokenize(self.source) + word_tokenize(self.answer)

if __name__ == '__main__':
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

    unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
    sentim_analyzer.add_feat_extractor(
        extract_unigram_feats, unigrams=unigram_feats)

    training_set = sentim_analyzer.apply_features(training_docs)
    test_set = sentim_analyzer.apply_features(testing_docs)

    trainer = NaiveBayesClassifier.train
    classifier = sentim_analyzer.train(trainer, training_set)
else:
    pass