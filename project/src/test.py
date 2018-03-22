import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re

def textToWords(inputString):
    letters_only = re.sub("[^a-zA-Z]", " ", str(inputString))
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return (" ".join(meaningful_words))

data = pd.read_csv("../data/fake.csv",sep=",")
data_text = data['text']
num_texts = data_text.size
clean_train_texts = []

for i in range( 0, num_texts ):
    if(i%1000==0):
        print("Processed ",i,"/",num_texts)
    clean_train_texts.append(textToWords(data_text[i]))

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

train_data_features = vectorizer.fit_transform(clean_train_texts)
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()
print (vocab)

#print(train_data_features[:10])

print(train_data_features.shape)