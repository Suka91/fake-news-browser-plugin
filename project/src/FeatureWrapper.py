import copy
import pandas as pd
from generators import BagOfWords, Punctations, SpellCheck, AverageLength, CapitalizedWords, ShallowSyntax
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
import os.path

class FeatureWrapper:
    def ScoreModel(self, train_data, test_data, commandsFile, useSVM=True, cv=True, outFile=None, saveModelPath=None, pcaOutputPath=None, transformerOutputPath=None):
        self.train_data = train_data
        self.test_data = test_data

        self.ss = ShallowSyntax()
        self.ss.transformData(train_data['text'], pcaOutputPath=pcaOutputPath)
        ss_train_features = copy.deepcopy(self.ss.shallow_syntax_features)


        self.sent_train = AverageLength()
        self.sent_train.transformData(train_data['text'])

        self.bow = BagOfWords()
        self.bow.transformData(train_data['text'], transformerOutputPath=transformerOutputPath)
        bow_train_features = copy.deepcopy(self.bow.data_features)

        self.punc_train = Punctations()
        self.punc_train.transformData(train_data['text'])

        self.ms_train = SpellCheck()
        self.ms_train.transformData(train_data['text'])

        self.cw_train = CapitalizedWords()
        self.cw_train.transformData(train_data['text'])

        if cv == False:
            self.ss.transformData(test_data['text'], trainModel=False)
            ss_test_features = copy.deepcopy(self.ss.shallow_syntax_features)

            self.sent_test = AverageLength()
            self.sent_test.transformData(test_data['text'])

            self.bow.transformData(test_data['text'], trainModel=False)
            bow_test_features = copy.deepcopy(self.bow.data_features)

            self.punc_test = Punctations()
            self.punc_test.transformData(test_data['text'])

            self.ms_test = SpellCheck()
            self.ms_test.transformData(test_data['text'])

            self.cw_test = CapitalizedWords()
            self.cw_test.transformData(test_data['text'])

        out_string = ""
        with open(commandsFile, 'r') as f:
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
                if (command == 'bow'):
                    new_features_train = pd.concat([new_features_train, pd.DataFrame(bow_train_features)],
                                                   axis=1)
                    if cv == False:
                        new_features_test = pd.concat([new_features_test, pd.DataFrame(bow_test_features)],
                                                      axis=1)
                elif (command == 'shallow_syntax'):
                    new_features_train = pd.concat(
                        [new_features_train, pd.DataFrame(ss_train_features)], axis=1)
                    if cv == False:
                        new_features_test = pd.concat(
                            [new_features_test, pd.DataFrame(ss_test_features)], axis=1)
                elif (command == 'misspelled_words'):
                    new_features_train = pd.concat([new_features_train, pd.DataFrame(self.ms_train.misspelled_freq)],
                                                   axis=1)
                    if cv == False:
                        new_features_test = pd.concat([new_features_test, pd.DataFrame(self.ms_test.misspelled_freq)],
                                                      axis=1)
                elif (command == 'punctation'):
                    new_features_train = pd.concat([new_features_train, pd.DataFrame(self.punc_train.punctuation_freq)],
                                                   axis=1)
                    if cv == False:
                        new_features_test = pd.concat([new_features_test, pd.DataFrame(self.punc_test.punctuation_freq)],
                                                      axis=1)
                elif (command == 'sentence'):
                    new_features_train = pd.concat([new_features_train, pd.DataFrame(self.sent_train.average_word_len),
                                                    pd.DataFrame(self.sent_train.average_sentence_len)], axis=1)
                    if cv == False:
                        new_features_test = pd.concat([new_features_test, pd.DataFrame(self.sent_test.average_word_len),
                                                       pd.DataFrame(self.sent_test.average_sentence_len)], axis=1)
                elif (command == 'capitalized_words'):
                    new_features_train = pd.concat(
                        [new_features_train, pd.DataFrame(self.cw_train.capitalized_words_freq)], axis=1)
                    if cv == False:
                        new_features_test = pd.concat(
                            [new_features_test, pd.DataFrame(self.cw_test.capitalized_words_freq)], axis=1)
                else:
                    print("Unkown command: ", command)

            if(useSVM == True):
                print("Begin fit SVM")
                clf = svm.SVC()
                clf.fit(new_features_train, self.train_data['type'])
            else:
                print("Define Logistic Regression model")
                clf = LogisticRegression(random_state=0).fit(new_features_train, self.train_data['type'])

            if (saveModelPath is not None):
                dump(clf, saveModelPath + '/' + title_string + '.joblib')

            print("Begin cross val")
            scores = cross_val_score(clf, new_features_train, self.train_data['type'], cv=5)
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
            with open(outFile, 'w') as f:
                f.write(out_string)


    def ScoreText(self, input_text, commands, saveModelPath, transformModelPath, pcaModelPath):
        df_data_text = pd.DataFrame([input_text], columns=['text'])
        new_features = None
        print("Begin ScoreText")
        commands_split = commands.split()
        for command in commands_split:
            if (command == 'bow'):
                bow = BagOfWords()
                bow.transformText(df_data_text['text'], transformModelPath)
                new_features = pd.concat([new_features, pd.DataFrame(bow.data_features)], axis=1)
            elif (command == 'shallow_syntax'):
                ss = ShallowSyntax()
                ss.transformText(df_data_text['text'], pcaModelPath)
                new_features = pd.concat([new_features, pd.DataFrame(ss.shallow_syntax_features)], axis=1)
            elif (command == 'misspelled_words'):
                ms = SpellCheck()
                ms.transformData(df_data_text['text'])
                new_features = pd.concat([new_features, pd.DataFrame(ms.misspelled_freq)], axis=1)
            elif (command == 'punctation'):
                punc = Punctations()
                punc.transformData(df_data_text['text'])
                new_features = pd.concat([new_features, pd.DataFrame(punc.punctuation_freq)], axis=1)
            elif (command == 'sentence'):
                sent = AverageLength()
                sent.transformData(df_data_text['text'])
                new_features = pd.concat([new_features, pd.DataFrame(sent.average_word_len), pd.DataFrame(sent.average_sentence_len)], axis=1)
            elif (command == 'capitalized_words'):
                cw = CapitalizedWords()
                cw.transformData(df_data_text['text'])
                new_features = pd.concat([new_features, pd.DataFrame(cw.capitalized_words_freq)], axis=1)
            else:
                print("Unkown command: ", command)

        clf = load(saveModelPath + '/' + commands + '.joblib')
        predicted = clf.predict(new_features)
        print("Predicted: " + str(predicted[0]))
        return predicted
    def DebugModel(self, transformModelPath):
        bow = BagOfWords()
        bow.debugGenerator(transformModelPath)
