import copy
import pandas as pd
import Consts
from generators import BagOfWords, Punctations, SpellCheck, AverageLength, CapitalizedWords, ShallowSyntax
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
import os.path

class FeatureWrapper:
    def ScoreModel(self, data_train, data_test, commandsFile, useSVM=True, outFile=None, saveModelPath=None, pcaOutputPath=None, transformerOutputPath=None):
        X_train = data_train['text']
        y_train = data_train['type']
        X_test = data_test['text']
        y_test = data_test['type']
        self.ss = ShallowSyntax()
        self.ss.transformData(X_train, pcaOutputPath=pcaOutputPath)
        ss_train_features = copy.deepcopy(self.ss.shallow_syntax_features)


        self.sent_train = AverageLength()
        self.sent_train.transformData(X_train)

        self.bow = BagOfWords()
        self.bow.transformData(X_train, transformerOutputPath=transformerOutputPath)
        bow_train_features = copy.deepcopy(self.bow.data_features)

        self.punc_train = Punctations()
        self.punc_train.transformData(X_train)

        self.ms_train = SpellCheck()
        self.ms_train.transformData(X_train)

        self.cw_train = CapitalizedWords()
        self.cw_train.transformData(X_train)

        self.ss.transformData(X_test, trainModel=False)
        ss_test_features = copy.deepcopy(self.ss.shallow_syntax_features)

        self.sent_test = AverageLength()
        self.sent_test.transformData(X_test)

        self.bow.transformData(X_test, trainModel=False)
        bow_test_features = copy.deepcopy(self.bow.data_features)

        self.punc_test = Punctations()
        self.punc_test.transformData(X_test)

        self.ms_test = SpellCheck()
        self.ms_test.transformData(X_test)

        self.cw_test = CapitalizedWords()
        self.cw_test.transformData(X_test)

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
                if (command == Consts.CM_BOW):
                    new_features_train = pd.concat([new_features_train,
                                                    pd.DataFrame(bow_train_features)],
                                                   axis=1)

                    new_features_test = pd.concat([new_features_test,
                                                   pd.DataFrame(bow_test_features)],
                                                  axis=1)
                elif (command ==  Consts.CM_SS):
                    new_features_train = pd.concat([new_features_train,
                                                    pd.DataFrame(ss_train_features)],
                                                   axis=1)
                    new_features_test = pd.concat([new_features_test,
                                                   pd.DataFrame(ss_test_features)],
                                                  axis=1)
                elif (command == Consts.CM_MW):
                    new_features_train = pd.concat([new_features_train,
                                                    pd.DataFrame(self.ms_train.misspelled_freq)],
                                                   axis=1)
                    new_features_test = pd.concat([new_features_test,
                                                   pd.DataFrame(self.ms_test.misspelled_freq)],
                                                  axis=1)
                elif (command == Consts.CM_PC):
                    new_features_train = pd.concat([new_features_train,
                                                    pd.DataFrame(self.punc_train.punctuation_freq)],
                                                   axis=1)
                    new_features_test = pd.concat([new_features_test,
                                                   pd.DataFrame(self.punc_test.punctuation_freq)],
                                                  axis=1)
                elif (command == Consts.CM_SL):
                    new_features_train = pd.concat([new_features_train,
                                                    pd.DataFrame(self.sent_train.average_word_len),
                                                    pd.DataFrame(self.sent_train.average_sentence_len)],
                                                   axis=1)
                    new_features_test = pd.concat([new_features_test,
                                                   pd.DataFrame(self.sent_test.average_word_len),
                                                   pd.DataFrame(self.sent_test.average_sentence_len)],
                                                  axis=1)
                elif (command == Consts.CM_CW):
                    new_features_train = pd.concat([new_features_train,
                                                    pd.DataFrame(self.cw_train.capitalized_words_freq)],
                                                   axis=1)
                    new_features_test = pd.concat([new_features_test,
                                                   pd.DataFrame(self.cw_test.capitalized_words_freq)],
                                                  axis=1)
                else:
                    print("Unkown command: ", command)

            if(useSVM == True):
                print("Begin fit SVM")
                clf = svm.SVC()
                clf.fit(new_features_train, y_train)
            else:
                print("Define Logistic Regression model")
                clf = LogisticRegression(random_state=Consts.LR_RANDOM_STATE,
                                         max_iter=Consts.LR_MAX_ITER).fit(new_features_train, y_train)

            if (saveModelPath is not None):
                dump(clf, saveModelPath + '/' + title_string + '.joblib')

            print("Begin cross val")
            scores = cross_val_score(clf, new_features_train, y_train, cv=Consts.CV_FOLDS)
            print(scores)
            print("Begin predict")
            acc_score = clf.score(new_features_test, y_test)
            print("End")
            print(str(acc_score))
            print(title_string)
            score_string = "cross_val: {d:.3f}".format(d = scores.mean()) + " / {d:.3f} ".format(d = scores.std()) + "\n" + "acc_test {d:.3f}:".format(d = acc_score) + "\n"
            print(score_string)
            out_string += "-----------------\n"
            out_string += title_string + "\n"
            out_string += score_string + "\n"
            out_string += "-----------------\n"
        if outFile is not None:
            with open(outFile, 'w') as f:
                f.write(out_string)


    def ScoreText(self, input_text, commands, modelPath, transformModelPath, pcaModelPath):
        df_data_text = pd.DataFrame([input_text], columns=['text'])
        new_features = None
        print("Begin ScoreText")
        commands_split = commands.split()
        for command in commands_split:
            if (command == Consts.CM_BOW):
                bow = BagOfWords()
                bow.transformText(df_data_text['text'], transformModelPath)
                new_features = pd.concat([new_features,
                                          pd.DataFrame(bow.data_features)],
                                         axis=1)
            elif (command == Consts.CM_SS):
                ss = ShallowSyntax()
                ss.transformText(df_data_text['text'], pcaModelPath)
                new_features = pd.concat([new_features,
                                          pd.DataFrame(ss.shallow_syntax_features)],
                                         axis=1)
            elif (command == Consts.CM_MW):
                ms = SpellCheck()
                ms.transformData(df_data_text['text'])
                new_features = pd.concat([new_features,
                                          pd.DataFrame(ms.misspelled_freq)],
                                         axis=1)
            elif (command == Consts.CM_PC):
                punc = Punctations()
                punc.transformData(df_data_text['text'])
                new_features = pd.concat([new_features,
                                          pd.DataFrame(punc.punctuation_freq)],
                                         axis=1)
            elif (command == Consts.CM_SL):
                sent = AverageLength()
                sent.transformData(df_data_text['text'])
                new_features = pd.concat([new_features,
                                          pd.DataFrame(sent.average_word_len),
                                          pd.DataFrame(sent.average_sentence_len)],
                                         axis=1)
            elif (command == Consts.CM_CW):
                cw = CapitalizedWords()
                cw.transformData(df_data_text['text'])
                new_features = pd.concat([new_features,
                                          pd.DataFrame(cw.capitalized_words_freq)],
                                         axis=1)
            else:
                print("Unkown command: ", command)

        clf = load(modelPath + '/' + commands + '.joblib')
        predicted = clf.predict(new_features)
        print("Predicted: " + str(predicted[0]))
        return predicted
    def DebugModel(self, transformModelPath):
        bow = BagOfWords()
        bow.debugGenerator(transformModelPath)
    def ScoreModelTest(self, data_train, commands, saveModelPath, transformModelPath):
        X_train = data_train['text']
        y_train = data_train['type']
        new_features = None
        bow = BagOfWords()
        bow.transformText(X_train, transformModelPath)
        new_features = pd.concat([new_features, pd.DataFrame(bow.data_features)], axis=1)
        clf = load(saveModelPath + '/' + commands + '.joblib')
        acc_score = clf.score(new_features, y_train)
        score_string = "acc_test {d:.3f}:".format(d=acc_score) + "\n"
        print(score_string)
