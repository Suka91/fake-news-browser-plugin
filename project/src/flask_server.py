import newspaper
from newspaper import Article
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS, cross_origin
from FeatureWrapper import FeatureWrapper

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
features = FeatureWrapper()
@app.route('/_predict/', methods=['POST'])
def _predict():
    print('Entered!')
    print(request.form['arg1'])
    article = Article(request.form['arg1'])
    article.download()
    article.parse()
    #print(article.text)

    # predicted = 0
    # for i in range(100,110):
    #     test_text = ""
    #     with open('../data/bbc/business/'+str(i)+'.txt', 'r') as f:
    #         test_text = f.read()
    #
    #     features = FeatureWrapper()
    #     predicted += features.ScoreText(test_text, "bow ", "../saved_models_1_2_ngrams_1500_features", pcaModelPath="../saved_models_1_2_ngrams_1500_features/pca.pickle",transformModelPath="../saved_models_1_2_ngrams_1500_features/tfidf.pickle")
    # print('TOTAL PREDICTED ' + str(predicted))
    predicted = features.ScoreText(article.text, "bow ", "../saved_models_1_2_ngrams_1500_features", pcaModelPath="../saved_models_1_2_ngrams_1500_features/pca.pickle", transformModelPath="../saved_models_1_2_ngrams_1500_features/tfidf.pickle")
    #myList = [str(predicted)]
    #print(request.data)

    print('Predicted: ' + str(predicted))
    return jsonify({'data' : str(predicted)})

if __name__ == "__main__":
    print('Start server!')
    predicted = features.DebugModel(transformModelPath="../saved_models_1_2_ngrams_1500_features/tfidf.pickle")
    app.run(debug=True)
