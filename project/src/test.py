import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re
from ast import literal_eval
import numpy as np

with open("../saved_models_1_2_ngrams_1500_features/output.txt") as f,\
     open("./Output.txt", 'w+') as o:
    lines = f.readlines()
    for line in lines:
        o.write(line)
        if(line.startswith("[")):
            a = re.sub('\s+', ',', line)
            a = np.array(literal_eval(a))
            print(a.std())
            print(a.mean())
            o.write(str(a.mean()) + " " + str(a.std()) + "\n")
        print("#########")
