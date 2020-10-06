import nltk
import random
import re
from nltk.corpus import movie_reviews, subjectivity
from nltk.sentiment.util import mark_negation, extract_unigram_feats

tokenizer = nltk.RegexpTokenizer(r"\w+")  # removes punctuation from tokens
porter = nltk.stem.PorterStemmer()


class Featurable:
    def __init__(self, featurable=''):
        self.string = featurable
        self.tokens = tokenizer.tokenize(featurable)
        self.ps_tag = nltk.pos_tag(self.tokens)
        self.nchunk = nltk.ne_chunk(self.ps_tag)
        self.ptstem = [porter.stem(token.lower()) for token in self.tokens]

    def get_char_count(self):
        return len(self.string)

    def get_word_count(self):
        return len(self.tokens)

    def get_words(self):
        return self.tokens

    def get_pos_tags(self):
        return self.ps_tag

    def get_ne_chunks(self):
        return self.nchunk

    def get_stems(self):
        return self.ptstem


class Sentence:
    def __init__(self, kwds={'source': "", 'question': "", 'answer': ""}):
        self.question = Featurable(kwds['question'])
        self.source = Featurable(kwds['source'])
        self.answer = Featurable(kwds['answer'])

    def get_all_words(self):
        all_stems = []
        all_stems.extend(self.question.get_stems())
        all_stems.extend(self.source.get_stems())
        all_stems.extend(self.answer.get_stems())
        return all_stems

    def extract_word_features(self, word_features):
        words = set(word_features)
        features = {}
        for w in word_features:
            features[w] = (w in words)

        return features
