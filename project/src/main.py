from generators import BagOfWords, Punctations, SpellCheck, AverageLength, CapitalizedWords, ShallowSyntax
from sklearn import svm
import pandas as pd
import os
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import random

class FeatureWrapper:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        
        self.ss_train = ShallowSyntax(train_data['text'])
        self.sent_train = AverageLength(train_data['text'])
        self.bow_train = BagOfWords(train_data['text'])
        self.punc_train = Punctations(train_data['text'])
        self.ms_train = SpellCheck(train_data['text'])
        self.cw_train = CapitalizedWords(train_data['text'])

        self.ss_test = ShallowSyntax(test_data['text'])
        self.sent_test = AverageLength(test_data['text'])
        self.bow_test = BagOfWords(test_data['text'])
        self.punc_test = Punctations(test_data['text'])
        self.ms_test = SpellCheck(test_data['text'])
        self.cw_test = CapitalizedWords(test_data['text'])

    def Score(self,commandsFile,outFile=None):
        out_string = ""
        new_features_train = pd.DataFrame()
        new_features_test = pd.DataFrame()
        with open(commandsFile,'r') as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        for commands in content:
            new_features_train = None
            new_features_test = None
            title_string = ""
            score_string = ""

            commands_split = commands.split()

            print(commands_split)
            for command in commands_split:
                title_string += command + " "
                if(command == 'bow'):
                    new_features_train = pd.concat([new_features_train,pd.DataFrame(self.bow_train.data_features)], axis=1)
                    new_features_test = pd.concat([new_features_test,pd.DataFrame(self.bow_test.data_features)], axis=1)
                elif(command == 'shallow_syntax'):
                    new_features_train = pd.concat([new_features_train,pd.DataFrame(self.ss_train.shallow_syntax_features)], axis=1)
                    new_features_test = pd.concat([new_features_test,pd.DataFrame(self.ss_test.shallow_syntax_features)], axis=1)
                elif(command == 'misspelled_words'):
                    new_features_train = pd.concat([new_features_train,pd.DataFrame(self.ms_train.misspelled_freq)], axis=1)
                    new_features_test = pd.concat([new_features_test,pd.DataFrame(self.ms_test.misspelled_freq)], axis=1)
                elif(command == 'punctation'):
                    new_features_train = pd.concat([new_features_train,pd.DataFrame(self.punc_train.punctuation_freq)], axis=1)
                    new_features_test = pd.concat([new_features_test,pd.DataFrame(self.punc_test.punctuation_freq)], axis=1)
                elif(command == 'sentence'):
                    new_features_train = pd.concat([new_features_train,pd.DataFrame(self.sent_train.average_word_len),pd.DataFrame(self.sent_train.average_sentence_len)], axis=1)
                    new_features_test = pd.concat([new_features_test,pd.DataFrame(self.sent_test.average_word_len),pd.DataFrame(self.sent_test.average_sentence_len)], axis=1)
                elif(command == 'capitalized_words'):
                    new_features_train = pd.concat([new_features_train,pd.DataFrame(self.cw_train.capitalized_words_freq)], axis=1)
                    new_features_test = pd.concat([new_features_test,pd.DataFrame(self.cw_test.capitalized_words_freq)], axis=1)
                else:
                    print("Unkown command: ",command)

            print("Begin fit SVM")
            clf = svm.SVC()
            clf.fit(new_features_train, self.train_data['type'])
            print("Begin cross val")
            scores = cross_val_score(clf, new_features_test, self.test_data['type'], cv=5)
            print(scores)
            print("Begin predict")
            predicted = clf.predict(new_features_train)
            print("End")
            acc_score = accuracy_score(self.train_data['type'], predicted)
            print(str(acc_score))
            print(title_string)
            score_string = str(scores) + "\n" + str(acc_score) + "\n"
            print(score_string)
            out_string += "-----------------\n"
            out_string += title_string + "\n"
            out_string += score_string + "\n"
            out_string += "-----------------\n"
        if outFile is not None:
            with open(outFile,'w') as f:
                f.write(out_string)


def readArticles(path):
    articles = []
    i = 0
    for dirName, subdirList, fileList in os.walk(path):
        for file in fileList:
            with open(dirName+"/"+file, 'r') as f:
                articles.append(f.read())
    return articles

random.seed(42)

data_true = readArticles("../data/bbc")
random.shuffle(data_true)
df_data_true = pd.DataFrame(data_true, columns=['text'])

data_false = pd.read_csv("../data/fake.csv",sep=",")
random.shuffle(data_false.as_matrix())
data_false = pd.DataFrame(data_false)

type_col = [0 for _ in range(0,len(data_false))]
data_false['type'] = type_col

type_col = [1 for _ in range(0,len(df_data_true))]
df_data_true['type'] = type_col

new_data = data_false.head(1482)
new_data = new_data[['text','type']]
train_data = new_data.append(df_data_true.head(1482), ignore_index=True)

new_data = data_false.tail(743)
new_data = new_data[['text','type']]
test_data = new_data.append(df_data_true.tail(743), ignore_index=True)

features = FeatureWrapper(train_data,test_data)
features.Score("../data/commands.txt","output.txt")
# ss_train = ShallowSyntax(train_data['text'])
# sent_train = AverageLength(train_data['text'])
# bow_train = BagOfWords(train_data['text'])
# punc_train = Punctations(train_data['text'])
# ms_train = SpellCheck(train_data['text'])
# cw_train = CapitalizedWords(train_data['text'])
#
# ss_test = ShallowSyntax(test_data['text'])
# sent_test = AverageLength(test_data['text'])
# bow_test = BagOfWords(test_data['text'])
# punc_test = Punctations(test_data['text'])
# ms_test = SpellCheck(test_data['text'])
# cw_test = CapitalizedWords(test_data['text'])
# new_features = pd.concat([pd.DataFrame(sent.average_sentence_len),\
#                           pd.DataFrame(sent.average_word_len),\
#                           pd.DataFrame(bow.data_features),\
#                           pd.DataFrame(punc.punctuation_freq),\
#                           pd.DataFrame(ms.misspelled_freq),\
#                           pd.DataFrame(cw.capitalized_words_freq),\
#                           pd.DataFrame(ss.shallow_syntax_features)], axis=1)
########
# out_string = ""
# new_features_train = pd.concat([pd.DataFrame(bow_train.data_features),pd.DataFrame(ss_train.shallow_syntax_features)], axis=1)
# new_features_test = pd.concat([pd.DataFrame(bow_test.data_features),pd.DataFrame(ss_test.shallow_syntax_features)], axis=1)
# print(new_features_train.shape)
# print(new_features_test.shape)
# clf = svm.SVC()
# print("Begin fit SVM")
# out_string += "-------------\n"
# out_string += "bow + shallow_syntax\n"
# out_string += "\n"
# out_string += "Begin fit SVM\n"
# clf.fit(new_features_train, train_data['type'])
# print("Begin cross val")
# out_string += "Begin cross val\n"
# scores = cross_val_score(clf, new_features_test, test_data['type'], cv=5)
# print(scores)
# out_string += str(scores)+"\n"
# print("Begin predict")
# out_string += "Begin predict\n"
# predicted = clf.predict(new_features_train)
# print("End")
# out_string += "End\n"
# print(accuracy_score(train_data['type'], predicted))
# out_string += str(accuracy_score(train_data['type'], predicted))+"\n"
#
# new_features_train = pd.concat([pd.DataFrame(bow_train.data_features),pd.DataFrame(punc_train.punctuation_freq)], axis=1)
# new_features_test = pd.concat([pd.DataFrame(bow_test.data_features),pd.DataFrame(punc_test.punctuation_freq)], axis=1)
# print(new_features_train.shape)
# print(test_data.shape)
# clf = svm.SVC()
# print("Begin fit SVM")
# out_string += "-------------\n"
# out_string += "bow + punctation\n"
# out_string += "\n"
# out_string += "Begin fit SVM\n"
# clf.fit(new_features_train, train_data['type'])
# print("Begin cross val")
# out_string += "Begin cross val\n"
# scores = cross_val_score(clf, new_features_test, test_data['type'], cv=5)
# print(scores)
# out_string += str(scores)+"\n"
# print("Begin predict")
# out_string += "Begin predict\n"
# predicted = clf.predict(new_features_train)
# print("End")
# out_string += "End\n"
# print(accuracy_score(train_data['type'], predicted))
# out_string += str(accuracy_score(train_data['type'], predicted))+"\n"
#
# new_features_train = pd.concat([pd.DataFrame(bow_train.data_features),pd.DataFrame(ms_train.misspelled_freq)], axis=1)
# new_features_test = pd.concat([pd.DataFrame(bow_test.data_features),pd.DataFrame(ms_test.misspelled_freq)], axis=1)
# print(new_features_train.shape)
# print(test_data.shape)
# clf = svm.SVC()
# print("Begin fit SVM")
# out_string += "-------------\n"
# out_string += "bow + misspelled words\n"
# out_string += "\n"
# out_string += "Begin fit SVM\n"
# clf.fit(new_features_train, train_data['type'])
# print("Begin cross val")
# out_string += "Begin cross val\n"
# scores = cross_val_score(clf, new_features_test, test_data['type'], cv=5)
# print(scores)
# out_string += str(scores)+"\n"
# print("Begin predict")
# out_string += "Begin predict\n"
# predicted = clf.predict(new_features_train)
# print("End")
# out_string += "End\n"
# print(accuracy_score(train_data['type'], predicted))
# out_string += str(accuracy_score(train_data['type'], predicted))+"\n"
#
# new_features_train = pd.concat([pd.DataFrame(bow_train.data_features),pd.DataFrame(sent_train.average_word_len),pd.DataFrame(sent_train.average_sentence_len)], axis=1)
# new_features_test = pd.concat([pd.DataFrame(bow_test.data_features),pd.DataFrame(sent_test.average_word_len),pd.DataFrame(sent_test.average_sentence_len)], axis=1)
# print(new_features_train.shape)
# print(test_data.shape)
# clf = svm.SVC()
# print("Begin fit SVM")
# out_string += "-------------\n"
# out_string += "bow + word len / sent len\n"
# out_string += "\n"
# out_string += "Begin fit SVM\n"
# clf.fit(new_features_train, train_data['type'])
# print("Begin cross val")
# out_string += "Begin cross val\n"
# scores = cross_val_score(clf, new_features_test, test_data['type'], cv=5)
# print(scores)
# out_string += str(scores)+"\n"
# print("Begin predict")
# out_string += "Begin predict\n"
# predicted = clf.predict(new_features_train)
# print("End")
# out_string += "End\n"
# print(accuracy_score(train_data['type'], predicted))
# out_string += str(accuracy_score(train_data['type'], predicted))+"\n"
# with open("output.txt", 'w') as f:
#     f.write(out_string)
#########