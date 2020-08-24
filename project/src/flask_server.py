import newspaper
from newspaper import Article
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS, cross_origin
from FeatureWrapper import FeatureWrapper

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
features = FeatureWrapper()
@app.route('/_get_data/', methods=['POST'])
def _get_data():
    print('Entered!')
    print(request.form['arg1'])
    article = Article(request.form['arg1'])
    article.download()
    article.parse()
    print(article.text)

    # predicted = 0
    # for i in range(100,110):
    #     test_text = ""
    #     with open('../data/bbc/business/'+str(i)+'.txt', 'r') as f:
    #         test_text = f.read()
    #
    #     features = FeatureWrapper()
    #     predicted += features.ScoreText(test_text, "bow ", "../saved_models", pcaModelPath="../saved_models/pca.pickle",transformModelPath="../saved_models/tfidf.pickle")
    # print('TOTAL PREDICTED ' + str(predicted))
    predicted = features.ScoreText(article.text, "bow ", "../saved_models", pcaModelPath="../saved_models/pca.pickle", transformModelPath="../saved_models/tfidf.pickle")
    #myList = [str(predicted)]
    #print(request.data)
    return jsonify({'data' : str(predicted)})

if __name__ == "__main__":
    app.run(debug=True)
