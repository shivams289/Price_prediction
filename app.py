from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import pandas as predict
from model import Model

app = Flask(__name__)
api = Api(app)

model = Model()
path = 'model/model.pkl'
with open(path, 'rb') as f:
    model.model = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictPrice(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        df = np.array([user_query])
        df = pd.DataFrame(df)
		df = model.preprocess(df)
		print('\nRunning Countvectorizer\n')
		X_name = model.CountVectorizer(df)
		print('\nRunning TFIDF\n')
		X_desc = model.TFIDFVectorizer(df)
		print('\nRunning LabelBinarizer\n')
		X_brand = model.Labelbinarizer(df)
		print('\nRunning Dummyencoders\n')
		X = model.Dummyencoder(df, X_name, X_desc, X_brand)

        # vectorize the user's query and make a prediction
        uq_vectorized = model.vectorizer_transform(np.array([user_query]))
        prediction = model.predict(uq_vectorized)

        # create JSON object
        output = {'price': prediction}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictPrice, '/')


if __name__ == '__main__':
    app.run(debug=True)