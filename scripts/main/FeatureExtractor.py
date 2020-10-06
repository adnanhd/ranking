import nltk, random, re
from nltk import pos_tag
from nltk import ne_chunk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews, subjectivity
from nltk.sentiment.util import mark_negation, extract_unigram_feats

class Featurable:
    def __init__(self, featurable=''):
        self.features = {}
        self.tokens = word_tokenize(featurable.lower())
        self.pos_tags = pos_tag(self.tokens)
        self.ne_chunk = ne_chunk(self.pos_tags)
        self.score = 0

        self.set_feature_value('__length__', len(featurable))
        self.set_feature_value('__wordct__', len(self.tokens))
        self.set_feature_value('__pos__', 0)
        self.set_feature_value('__neu__', 0)
        self.set_feature_value('__neg__', 0)

        porter = PorterStemmer()
        for ((token, pos), ner) in self.ne_chunk.pos():
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

    def get_features(self, feature_set):
        return self.features

    def get_pos_tags(self):
        return self.pos_tags

    def add_num_of_vag_nps(self):
        pattern = "NP !<< PP|SBAR|ADVP " \
				+ "!<< (CD|NNS < /^\\d\\d\\d\\ds?$/) " \
				+ "!<< (NP < POS) " + "!< NNP|NNPS " \
                + "[ !> NP | < POS ]"

        x = re.search(pattern,"NP !<< PP|SBAR|ADVP") # [('word','POS_TAG')] -> 'POS_TAG'
        print(x)
        if x: self.inc_feature_value('__num_of_vag_nps__')
        self.score = (self.get_feature_value("__num_of_vag_nps__") * - 0.3 + self.get_feature_value('__length__') * 0.2) % 5

    def print_features(self):
        print(self.features)
        print('score = {0}'.format(self.score))


class Sentence:
    def __init__(self, kwds={'source': "", 'question': "", 'answer': ""}):
        self.question = kwds['question']
        self.source = kwds['source']
        self.answer = kwds['answer']

        self.question_feature = Featurable(self.question)
        self.source_feature = Featurable(self.source)
        self.answer_feature = Featurable(self.answer)

    def get_question(self):
        return self.question

    def get_source(self):
        return self.source

    def get_answer(self):
        return self.answer

    def get_words(self):
        return word_tokenize(self.question) + word_tokenize(self.source) + word_tokenize(self.answer)

    def get_feature_vector(self, feature_dict):
        return -1

if __name__ == '__main__':
    f = Featurable('Jhon\'s book was red')
    f.add_num_of_vag_nps()
    f.print_features()
