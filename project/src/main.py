import pandas as pd
import os
import random
import Consts
import numpy as np
from FeatureWrapper import FeatureWrapper
from sklearn.model_selection import train_test_split


def readArticles(path):
    articles = []
    i = 0
    for dirName, subdirList, fileList in os.walk(path):
        for file in fileList:
            with open(dirName+"/"+file, 'rb') as f:
                temp = f.read()
                articles.append(temp)
                i = i + 1
    return articles

random.seed(42)

print("Begin reading data..")

#Used to read BBC articles
# data_true = readArticles("../data/bbc")
# random.shuffle(data_true)
# df_data_true = pd.DataFrame(data_true, columns=['text'])
#
# type_col = [1 for _ in range(0,len(df_data_true))]
# df_data_true['type'] = type_col
#------------------------
#Used to read fake csv dataset
# data_false = pd.read_csv("../data/fake.csv",sep=",")
# random.shuffle(data_false.as_matrix())
# data_false = pd.DataFrame(data_false)
#
# type_col = [0 for _ in range(0,len(data_false))]
# data_false['type'] = type_col
#------------------------
#Create train data
# new_data = data_false.head(2225)
# new_data = new_data[['text','type']]
# data = new_data.append(df_data_true.head(2225), ignore_index=True)
#------------------------

data = pd.read_csv("../data/news_datasets.csv",sep=",", nrows = 100)

data = data[['text','type']]
data['text'] = data['text'].str.replace(r'^ *$', '')
data['text'].replace("", np.nan, inplace=True)
data.dropna(subset=['text'], inplace=True)
data.reset_index(drop=True, inplace=True)

data.loc[data['type'] == 'FAKE', 'type'] = 0
data.loc[data['type'] == 'REAL', 'type'] = 1
data['type'] = data['type'].astype('int')
#------------------------

print("Begin split data..")
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
data_train.to_csv("train.csv", sep=',', index=False)
data_test.to_csv("test.csv", sep=',', index=False)

data_train = pd.read_csv("train.csv",sep=",")
data_test = pd.read_csv("test.csv",sep=",")
os.remove("train.csv")
os.remove("test.csv")
print("Begin creating model..")
features = FeatureWrapper()
features.ScoreModel(data_train, data_test, Consts.INPUT_COMMANDS_PATH,
                    useSVM=True,
                    outFile=Consts.OUTPUT_DIR_PATH + Consts.OUTPUT_FILE_PATH,
                    saveModelPath=Consts.OUTPUT_DIR_PATH,
                    pcaOutputPath=Consts.OUTPUT_DIR_PATH + Consts.PCA_PICKLE_PATH,
                    transformerOutputPath=Consts.OUTPUT_DIR_PATH + Consts.TFIDF_PICKLE_PATH)

# features_lr = FeatureWrapper()
# features_lr.ScoreModel(data_train, data_test, "../data/commands.txt", useSVM=False, outFile="../models/z_saved_models_lr_1_2_ngrams_3000_features/output_lr1.txt")

# features = FeatureWrapper()
# features.ScoreModelTest(data, Consts.INPUT_COMMANDS, Consts.INPUT_DIR_PATH, transformModelPath=Consts.INPUT_DIR_PATH + Consts.TFIDF_PICKLE_PATH)



# ss_train = ShallowSyntax(train_data['text'])
# sent_train = AverageLength(train_data['text'])
# bow_train = BagOfWords(train_data['text'])
# punc_train = Punctations(train_data['text'])
# ms_train = SpellCheck(train_data['text'])
# cw_train = CapitalizedWords(train_data['text'])
#
# ss_test = ShallowSyntax(test_data['text'])
# sent_test = AverageLength(test_data['text'])
# bow_test = BagOfWords(test_data['text'])/
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