#Commands
CM_BOW = 'bow'
CM_SS = 'shallow_syntax'
CM_MW = 'misspelled_words'
CM_PC = 'punctation'
CM_SL = 'sentence'
CM_CW = 'capitalized_words'

#Logistic regression constants
LR_RANDOM_STATE = 0
LR_MAX_ITER = 500

#Cross valididation constants
CV_FOLDS = 5

#Debug constants
ITER_PRINT = 500

#TFIDF constants
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_DF = 0.95
TFIDF_MIN_DF = 2
TFIDF_MAX_FEATURES = 3000

#Punctations constants
PC_MUL = 500
PC_POW = 2
PUNCT_MARKS = '".,;!-'

#SpellCheck constants
SC_MUL = 100
SC_POW = 2

#AverageLength constants
AL_POW = 2

#CapitalizedWords constants
CW_MIN_WORD_LEN = 4
CW_MUL = 100
CW_POW = 2
CW_NEG_MUL = -0.001

#ShallowSyntax constants
SS_MUL = 100
SS_POW = 2

#PCA
PCA_COMPONENTS = 0.6

#Input dir
INPUT_COMMANDS = "bow "
INPUT_DIR_PATH = "../saved_models_1_2_ngrams_1500_features"
PCA_PICKLE_PATH = "/pca.pickle"
TFIDF_PICKLE_PATH = "/tfidf.pickle"

#Training dirs
INPUT_COMMANDS_PATH = "../data/commands.txt"
OUTPUT_DIR_PATH = "../models/n_saved_models_1_2_ngrams_3000_features"
OUTPUT_FILE_PATH = "/output.txt"
