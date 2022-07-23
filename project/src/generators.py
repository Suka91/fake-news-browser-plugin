import nltk
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import pandas as pd
import numpy as np
import enchant
from sklearn.decomposition import PCA
import pickle

class BagOfWords:
    def transformData(self, data, trainModel=True, transformerOutputPath = None):
        self.loadNltk()
        num_texts = data.size
        clean_train_texts = []

        for i in range(0, num_texts):
            if (i % 500 == 0):
                print("BagOfWords: Processed ", i, "/", num_texts)
            clean_train_texts.append(self.textToWords(data[i]))

        '''
        vectorizer = CountVectorizer(analyzer="word", \
                                     tokenizer=None, \
                                     preprocessor=None, \
                                     stop_words=None, \
                                     max_features=3000, \
                                     max_df=0.95, min_df=2, \
                                     ngram_range=(1, 2)
                                     )
        '''
        if(trainModel == True):
            vectorizer = TfidfVectorizer(analyzer='word',
                                         sublinear_tf=True,
                                         strip_accents='unicode',
                                         ngram_range=(1, 2),
                                         max_df=0.95, min_df=2,
                                         max_features=1500)
            tfidf = vectorizer.fit(clean_train_texts)
            if(transformerOutputPath is not None):
                pickle.dump(tfidf, open(transformerOutputPath, "wb"))
            data_features = vectorizer.transform(clean_train_texts).toarray()

            self.vectorizer = vectorizer
        else:
            data_features = self.vectorizer.transform(clean_train_texts).toarray()

        self.data_features = data_features
        self.feature_names = self.vectorizer.get_feature_names()
    '''
        tfidf_transformer = TfidfTransformer()
        data_features_tfidf = tfidf_transformer.fit_transform(data_features).toarray()

        self.tfidf_transformer = tfidf_transformer
        self.data_features_tfidf = data_features_tfidf
    '''
    def transformText(self, textData, transformerModelPath):
        self.loadNltk()
        num_texts = textData.size
        clean_text = []
        for i in range(0, num_texts):
            if (i % 500 == 0):
                print("BagOfWords: Processed ", i, "/", num_texts)
            clean_text.append(self.textToWords(textData[i]))


        # tfidf.pickle
        self.vectorizer = pickle.load(open(transformerModelPath, "rb"))
        self.data_features = self.vectorizer.transform(clean_text).toarray()

    def textToWords(self,inputString):
        letters_only = re.sub("[^a-zA-Z]", " ", str(inputString))
        words = letters_only.lower().split()
        stops = set(nltk.corpus.stopwords.words("english"))
        meaningful_words = [w for w in words if not w in stops]
        return (" ".join(meaningful_words))

    def loadNltk(self):
        try:
            nltk.data.find('help/stopwords')
        except LookupError:
            nltk.download('stopwords')

    def debugGenerator(self, transformerModelPath):
        self.vectorizer = pickle.load(open(transformerModelPath, "rb"))
        print(self.vectorizer.get_feature_names())

class Punctations:
    def transformData(self, data):
        num_texts = data.size
        punctuation_freq = []
        count = lambda l1, l2: sum([1 for x in l1 if x in l2])
        for i in range(0, num_texts):
            if (i % 500 == 0):
                print("Punctations: Processed ", i, "/", num_texts)
            punctuation_freq.append((500*(count(data[i], set(string.punctuation))/len(data[i])))**2)
        self.punctuation_freq = pd.DataFrame(np.array(punctuation_freq),columns=['punctuation_frequency'])

class SpellCheck:
    def transformData(self, data):
        num_texts = data.size
        misspelled_freq = []
        d = enchant.Dict("en_US")
        for i in range(0, num_texts):
            if (i % 500 == 0):
                print("SpellCheck: Processed ", i, "/", num_texts)
            letters_only = re.sub("[^a-zA-Z]", " ", str(data[i]))
            words = letters_only.lower().split()
            ms = sum([1 for x in words if d.check(x) is False])
            if(ms != 0):
                freq = (100*ms/len(words))**2
            else:
                freq = 0
            misspelled_freq.append(freq)
        self.misspelled_freq = pd.DataFrame(np.array(misspelled_freq), columns=['misspelled_freq'])

class AverageLength:
    def transformData(self, data):
        num_texts = data.size
        avg_sent_len = []
        avg_word_len = []
        for i in range(0, num_texts):
            if (i % 500 == 0):
                print("AverageLength: Processed ", i, "/", num_texts)
            filtered = ''.join(filter(lambda x: x not in '".,;!-', str(data[i])))
            words = [word for word in filtered.split() if word]

            avg_word_len.append((sum(map(len, words)) / len(words))**2)

            sents = str(data[i]).split('.');
            filtered_sents = []
            for sent in sents:
                sent = ''.join(filter(lambda x: x not in '".,;!-', sent))
                filtered_sents.append(sent)
            avg_sent_len.append((sum(map(len, filtered_sents)) / len(filtered_sents))**2)

        self.average_word_len = pd.DataFrame(np.array(avg_word_len), columns=['average_word_len'])
        self.average_sentence_len = pd.DataFrame(np.array(avg_sent_len), columns=['average_sentence_len'])

class CapitalizedWords:
    def transformData(self, data):
        cap_freq = []
        num_texts = data.size
        for i in range(0, num_texts):
            if (i % 500 == 0):
                print("CapitalizedWords: Processed ", i, "/", num_texts)
            filtered = ''.join(filter(lambda x: x not in '".,;!-', str(data[i])))
            filtered = filtered.split()
            freq = (100*len([word for word in filtered if word.isupper() and len(word)>4])/len(filtered))**2
            if(freq == 0):
                freq = -1*len(filtered)*0.001
            cap_freq.append(freq)

        self.capitalized_words_freq = pd.DataFrame(np.array(cap_freq), columns=['capitalized_words_freq'])

class ShallowSyntax:
    def transformData(self, data, trainModel=True, pcaOutputPath = None):
        self.loadNltk()
        num_texts = data.size
        columns = []
        tagdict = nltk.load('help/tagsets/upenn_tagset.pickle')
        for key in tagdict.keys():
            columns.append(str(key))

        df = pd.DataFrame(columns=columns)

        for i in range(0, num_texts):
            if (i % 500 == 0):
                print("ShallowSyntax: Processed ", i, "/", num_texts)

            new_row = pd.DataFrame(index = [i],columns = columns)
            for key in tagdict.keys():
                new_row.set_value(i, key, 0)

            text = nltk.tokenize.word_tokenize((str(data[i]).replace(r'"(.*?)"', '')))
            tagged_text = nltk.pos_tag(text)
            tag_fd = nltk.FreqDist(tag for (word, tag) in tagged_text)
            for (key,value) in tag_fd.items():
                if(key in tagdict.keys()):
                    new_row.set_value(i, key, 100*(value/tag_fd.N())**2)
            df = pd.concat([df,new_row])
        if(trainModel == True):
            pca = PCA(n_components=0.6,svd_solver='full')
            pca.fit(df.values)
            if(pcaOutputPath is not None):
                #pca.pickle
                pickle.dump(pca, open(pcaOutputPath, "wb"))
            transformed_features = pca.transform(df.values)
            self.pca = pca
        else:
            transformed_features = self.pca.transform(df.values)

        self.shallow_syntax_features = pd.DataFrame(transformed_features)

    def transformText(self, textData, pcaModelPath):
        self.loadNltk()
        num_texts = textData.size
        columns = []
        tagdict = nltk.load('help/tagsets/upenn_tagset.pickle')
        for key in tagdict.keys():
            columns.append(str(key))

        df = pd.DataFrame(columns=columns)

        for i in range(0, num_texts):
            if (i % 500 == 0):
                print("ShallowSyntax: Processed ", i, "/", num_texts)

            new_row = pd.DataFrame(index=[i], columns=columns)
            for key in tagdict.keys():
                new_row.set_value(i, key, 0)

            text = nltk.tokenize.word_tokenize((str(textData[i]).replace(r'"(.*?)"', '')))
            tagged_text = nltk.pos_tag(text)
            tag_fd = nltk.FreqDist(tag for (word, tag) in tagged_text)
            for (key, value) in tag_fd.items():
                if (key in tagdict.keys()):
                    new_row.set_value(i, key, 100 * (value / tag_fd.N()) ** 2)
            df = pd.concat([df, new_row])

        self.pca = pickle.load(open(pcaModelPath, "rb"))

        transformed_features = self.pca.transform(df.values)

        self.shallow_syntax_features = pd.DataFrame(transformed_features)

    def loadNltk(self):
        try:
            nltk.data.find('help/tagsets')
        except LookupError:
            nltk.download('tagsets')
        try:
            nltk.data.find('help/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('help/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')