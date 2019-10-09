import nltk
import random
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy


def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})


pos = []
with open("./pos_tweets.txt") as file:
    for line in file: 
        pos.append([line, 'pos'])
        
neg = []
with open("./neg_tweets.txt") as file:
    for line in file: 
        neg.append([line, 'neg'])



def preprocess():
    pass

# Generate 10 example
import copy

training_pre = pos[:len(pos)-5] + neg[:len(neg)-5]
test_pre = pos[len(pos)-5:] + neg[len(neg)-5:]

training = []
test = []
for k, line in enumerate(training_pre):
    training.append([format_sentence(line[0]), line[1]])

for k, line in enumerate(test_pre):
    test.append([format_sentence(line[0]), line[1]])

# Build classfier
from nltk.classify import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(training)

classifier.show_most_informative_features()

# Neg example
example1 = "XBox Live still down "
result = classifier.classify(format_sentence(example1))
print(result)


example1 = "wow... all i wanted to do is see taylor swift but of course shes sold out! "
result = classifier.classify(format_sentence(example1))
print(result)


example1 = "very sad after cavs loss "
result = classifier.classify(format_sentence(example1))
print(result)

example1 = "watching the cavs get their butts kicked by orlando "
result = classifier.classify(format_sentence(example1))
print(result)


example1 ="this week is not going as i had hoped "
result = classifier.classify(format_sentence(example1))
print(result)

# Pos example
example1 = " just watched the movie Wanted... it was pretty darn good."
result = classifier.classify(format_sentence(example1))
print(result)


example1 = "@chasepino I wish I was as cool as you.. "
result = classifier.classify(format_sentence(example1))
print(result)


example1 = "@DrSecret Nice to meet you too buddy "
result = classifier.classify(format_sentence(example1))
print(result)


example1 = "@FoxxFiles nah...The Cavs are done  Go Magic!"
result = classifier.classify(format_sentence(example1))
print(result)


example1 = "@DJTinaSapp Ha! I'll get it back to you as soon as possible! "
result = classifier.classify(format_sentence(example1))
print(result)

# Accurancy

from nltk.classify.util import accuracy
print(accuracy(classifier, test))

