from newspaper import Article
import Consts
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
    predicted = features.ScoreText(article.text, Consts.INPUT_COMMANDS, Consts.INPUT_DIR_PATH, pcaModelPath=Consts.INPUT_DIR_PATH + Consts.PCA_PICKLE_PATH, transformModelPath=Consts.INPUT_DIR_PATH + Consts.TFIDF_PICKLE_PATH)

    print('Predicted: ' + str(predicted))
    return jsonify({'data' : str(predicted)})

if __name__ == "__main__":
    print('Start server!')
    predicted = features.DebugModel(transformModelPath=Consts.INPUT_DIR_PATH + Consts.TFIDF_PICKLE_PATH)
    app.run(debug=True)
