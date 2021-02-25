import os
import pickle
from keras import models
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def decode_sentiment(score):
    return "Positive" if score > 0.5 else "Negative"

def predict_tweet(tweet_text):

    MAX_SEQUENCE_LENGTH = 30
    # preprocessing step (tweet conversion using word embedding)
    # loading
    with open(os.path.join(settings.HOME_DIRECTORY, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1
    print("Vocabulary Size :", vocab_size)

    from keras.preprocessing.sequence import pad_sequences

    x_test = pad_sequences(tokenizer.texts_to_sequences(tweet_text),
                           maxlen=MAX_SEQUENCE_LENGTH)

    model = models.load_model(os.path.join(settings.HOME_DIRECTORY, "model_network"))
    model.load_weights(os.path.join(os.path.join(settings.HOME_DIRECTORY, 'checkpoint'), 'weights.ckpt'))

    predictions = model.predict(x_test)

    y_pred_1d = [decode_sentiment(score) for score in predictions]
    y_true = x_test
    y_pred_numeric = []
    for y in y_pred_1d:
        if y == "Negative":
            y_pred_numeric.append(0)
        else:
            y_pred_numeric.append(1)
    target_names = ['positive', 'negative']
    print(classification_report(y_true, y_pred_numeric, target_names=target_names))
    return predictions



############################### API ################################

from flask import Flask, Blueprint, request, jsonify, make_response, abort

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False #this is cause otherwise JSON's labels are sorted alphabetically

import settings


def configure_app(flask_app):
    flask_app.config['SERVER_NAME'] = settings.FLASK_SERVER_NAME
    flask_app.config['SWAGGER_UI_DOC_EXPANSION'] = settings.RESTPLUS_SWAGGER_UI_DOC_EXPANSION
    flask_app.config['RESTPLUS_VALIDATE'] = settings.RESTPLUS_VALIDATE
    flask_app.config['RESTPLUS_MASK_SWAGGER'] = settings.RESTPLUS_MASK_SWAGGER
    flask_app.config['ERROR_404_HELP'] = settings.RESTPLUS_ERROR_404_HELP


def initialize_app(flask_app):
    configure_app(flask_app)
    blueprint = Blueprint('api', __name__, url_prefix='/flaskAPI')
    api.init_app(blueprint)
    api.add_namespace(ns_endpoint)

    flask_app.register_blueprint(blueprint)


def avg(arr):
    res = 0
    for x in arr:
        res += x
    res /= len(arr)
    return res

#this function transform a row of a database in a dictionary type data
def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

#-------------------------------------HANDLERS--------------------------------------------#

# this handles the errors for missing parameters
@app.errorhandler(400)
def bad_request(error):
    if not request.json:
        parameter = 's' # to print the plural (Missing input parameterS)
    else:
        for req in request.json:
            if request.path.startswith('/api/v1.0/sentimentText') and 'text' not in dict(req).keys():
                parameter = ' [\'text\']'
            elif request.path.startswith('/api/v1.0/sentimentKeyword') and 'keyword' not in request.json:
                parameter = ' [\'keyword\']'
            elif request.path.startswith('/api/v1.0/sentimentUser') and 'user_id' not in request.json:
                parameter = ' [\'user_id\']'
    status = 'STATUS.BAD_REQUEST'
    code = 400
    message = 'Missing input parameter{}'.format(parameter)
    return make_response(jsonify(
                            {
                                'status' : status,
                                'code' : code,
                                'message' : message
                            }), 400)

# this handles errors for incorrect format
@app.errorhandler(401)
def input_format(error):
    status = 'STATUS.INPUT_FORMAT_ERROR'
    code = 401
    message = 'Incorrect input parameter'
    return make_response(jsonify(
                            {
                                'status' : status,
                                'code' : code,
                                'message' : message
                            }), 401)

# this is for missing data errors
@app.errorhandler(404)
def not_found(error):
    status = 'STATUS.NOT_FOUND'
    code = 402
    message = 'Data not found'
    return make_response(jsonify(
                            {
                                'status' : status,
                                'code' : code,
                                'message' : message
                            }), 402)

#----------------------------------------------------------------------------------------------------#

@app.route('/api/v1.0/sentimentText', methods=['GET'])
def api_Text():

    status = 'STATUS.OK'
    code = 200
    data = []

    if not request.json:
        abort(400)

    text = []
    result = {'tweet_id': '', 'positive': '', 'negative': ''}

    for req in request.json:
        if 'text' not in dict(req).keys():
            abort(400)
        else:
            text.append(dict(req)['text'])

        if 'tweet_id' in dict(req).keys():
            result['tweet_id'] = dict(req)['tweet_id']

        data.append(result)

    predicted_sentiment_array = predict_tweet(text)
    for i in range(len(predicted_sentiment_array)):
        positive = round(1 - predicted_sentiment_array[i][0], 8)
        negative = round(predicted_sentiment_array[i][0], 8)

        data[i]['positive'] = str(positive)
        data[i]['negative'] = str(negative)

    return jsonify({'status' : status,
                    'code' : code,
                    'data' : data})



@app.route('/api/v1.0/sentimentKeyword', methods=['GET'])
def api_Keyword():

    status = 'STATUS.OK'
    code = 200

    if not request.json or not 'keyword' in request.json:
        abort(400)

    keyword = request.json['keyword']
    if type(keyword) != str:
        abort(401)

    df = pd.read_csv(settings.HOME_DIRECTORY + settings.DATASET_FILE,
                     encoding='latin', header=None)

    df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']
    df = df.drop(['sentiment','id', 'date', 'query', 'user_id'], axis=1)

    tweet_result = []

    tweet_set = df["text"].array

    for tweet in tweet_set:
        if tweet.find(keyword) != -1:
            tweet_result.append(tweet)

    if tweet_result == []:
        abort(404)

    predicted_sentiment_array = predict_tweet(tweet_result)
    predicted_sentiment = avg(predicted_sentiment_array)
    positive = round(1-predicted_sentiment[0],8)
    negative = round(predicted_sentiment[0],8)
    data = {'keyword': keyword, 'positive': str(positive), 'negative': str(negative)}

    return jsonify({'status' : status,
                    'code' : code,
                    'data' : data})


@app.route('/api/v1.0/sentimentUser', methods=['GET'])
def api_User():

    status = 'STATUS.OK'
    code = 200

    if not request.json or not 'user_id' in request.json:
        abort(400)

    user_id = request.json['user_id']
    if type(user_id) != str:
        abort(401)

    df = pd.read_csv(settings.HOME_DIRECTORY + settings.DATASET_FILE,
                     encoding='latin', header=None)

    df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']
    df = df.drop(['sentiment','id', 'date', 'query'], axis=1)

    tweet_result = []

    tweet_set = df["text"].array
    user_id_set = df["user_id"].array

    for i in range(len(user_id_set)):
        if user_id != user_id_set[i]:
            continue
        tweet_result.append(tweet_set[i])

    if tweet_result == []:
        abort(404)

    predicted_sentiment_array = predict_tweet(tweet_result)
    predicted_sentiment = avg(predicted_sentiment_array)
    positive = round(1-predicted_sentiment[0],8)
    negative = round(predicted_sentiment[0],8)
    data = {'user_id': user_id, 'positive': str(positive), 'negative': str(negative)}

    return jsonify({'status' : status,
                    'code' : code,
                    'data' : data})


if __name__ == '__main__':

    #initialize_app(app)
    app.run(debug= settings.FLASK_DEBUG)