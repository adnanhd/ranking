import nltk
import string
import math

tokenizer = nltk.RegexpTokenizer(r"\w+")  # removes punctuation from tokens
porter = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()


class Featurable:
    def __init__(self, featurable='', stopwords=()):
        self.string = featurable
        self.tokens = [token.lower() for token in tokenizer.tokenize(featurable) if len(token) > 0 and
                       token not in string.punctuation and token.lower() not in stopwords]
        self.ps_tag = nltk.pos_tag(self.tokens)
        self.nchunk = nltk.ne_chunk(self.ps_tag)
        self.features = {}

    def get_char_count(self):
        return len(self.string)

    def get_word_count(self):
        return len(self.tokens)

    def get_words(self):
        return self.tokens

    def get_pos_tags(self):
        return [pos for (token, pos) in self.ps_tag]

    def get_ne_chunks(self):
        return [nec for (token, nec) in self.nchunk]

    def get_stems(self):
        return [porter.stem(token.lower()) for token in self.tokens]

    def get_cleaned_tokens(self, stopwords=()):
        return [lemmatizer.lemmatize(token, 'n'
                                     if tag.startswith('NN') else 'v'
                                     if tag.startswith('VB') else 'a')
                for (token, tag) in self.ps_tag]

    def get_features(self):
        if not self.features:
            lemmas = [lemmatizer.lemmatize(token, 'n'
                                           if tag.startswith('NN') else 'v'
                                           if tag.startswith('VB') else 'a')
                      for (token, tag) in self.ps_tag]
            pos = [pos for (token, pos) in self.ps_tag]
            ner = [ner for (token, ner) in self.nchunk]

            self.features = dict(nltk.probability.FreqDist(lemmas + pos + ner))

            self.features['LEN__'] = math.log10(len(self.string))

            for i in range(5, 51, 5):
                if i < len(self.string):
                    self.features['LEN_' + str(i) + '_'] = 1.0
                else:
                    break

            self.features['CNT__'] = math.log2(len(self.tokens))
        return self.features


class Sentence:
    def __init__(self, kwds={'source': "", 'question': "", 'answer': ""}):
        self.question = Featurable(kwds['question'])
        self.source = Featurable(kwds['source'])
        self.answer = Featurable(kwds['answer'])
        self.morph_feat_dict = {}

    def get_all_words(self):
        all_words = []
        all_words.extend(self.question.get_lemmas())
        all_words.extend(self.source.get_lemmas())
        all_words.extend(self.answer.get_lemmas())
        return all_words

    def extract_morphological_features(self):  # lemmas pos ner
        if not self.morph_feat_dict:
            feature_dicts = []
            feature_dicts.append((self.question.get_features(), 'que.'))
            feature_dicts.append((self.source.get_features(), 'src.'))
            feature_dicts.append((self.answer.get_features(), 'ans.'))

            self.morph_feat_dict = {prefix+key: feat_dict[key] for (
                feat_dict, prefix) in feature_dicts for key in feat_dict.keys()}

        return self.morph_feat_dict

    def extract_vagueness_features(self):
        return 0


if __name__ == '__main__':
    stops = tuple([word for word in nltk.corpus.stopwords.words('english')
                   if not word.startswith('wh') or word.startswith('how')])
    test = {'question': "Who are interns in enocta?",
            'source': "Adnan and Asrin are interns in Enocta Technologies in Asia",
            'answer': "Adnan and Asrin"}

    wf = Featurable(test['question'] + test['source'] + test['answer'])
    s = Sentence(test)
    print(s.extract_morphological_features())
