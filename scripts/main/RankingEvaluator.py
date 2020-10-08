import nltk
import random
import sklearn
from sklearn.svm import LinearSVC
from FeatureExtractor import Sentence


class RankingEvaluator:
    # [({'source':"", 'question':"", 'answer':""}, <LABEL>)]
    def __init__(self):
        self.classifier = nltk.classify.SklearnClassifier(
            LinearSVC(max_iter=5000))

    def fit(self, corpus):
        # corpus: [({'question':'str','answer':'str','source':'str'}, 'label')]
        dataset = [(Sentence(data).extract_morphological_features(), label)
                   for (data, label) in corpus]

        self.classifier.train(dataset)
        print('model trained...')

    def predict(self, instance):
        instance = Sentence(instance)
        res = self.classifier.classify(
            instance.extract_morphological_features())
        print('data predicted to be {0}...'.format(res))
        return res

    def test(self, tests):
        testset = [(Sentence(test_data).extract_morphological_features(), label)
                   for (test_data, label) in tests]
        predict = [self.classifier.classify(
            sentence) == label for (sentence, label) in testset]
        print("Data tested...\nAccuracy: {0}".format(
            sum(predict)/len(predict)))


if __name__ == '__main__':
    positive_samples = open('model/positive_samples.txt')
    negative_samples = open('model/negative_samples.txt')
    alperens_samples = open('tests/alperens_samples.txt')

    corpus = []
    corpus.extend([(eval(line), True)
                   for line in positive_samples.readlines()])
    corpus.extend([(eval(line), False)
                   # line {'question':'str','answer':'str','source':'str'}
                   for line in negative_samples.readlines()])
    test2_set = [(eval(line), True) for line in alperens_samples.readlines()]

    random.shuffle(corpus)
    train_set = corpus[:int(len(corpus)*.8)]
    tests_set = corpus[int(len(corpus)*.8):]

    positive_samples.close()
    negative_samples.close()

    re = RankingEvaluator()
    re.fit(train_set)
    re.test(tests_set)

    for line in test2_set:
        print("Source: {0}\nQuestion: {1}\nAnswer: {2}\nCategory: {3}\n\n\n".format(
            line[0]['source'], line[0]['question'], line[0]['answer'], line[1]))

    q = ' '
    a = ' '
    s = ' '
    while q != '' and a != '' and s != '':
        q = input('question> ')
        a = input('answer> ')
        s = input('source> ')
        print('\n')

        re.predict({'question': q, 'answer': a, 'source': s})
        print('\n\n')
