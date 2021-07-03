import json
import logging
import plotly
import pandas as pd


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib 
from sqlalchemy import create_engine

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
#import sklearn.ensemble.forest

from train_classifier import DisasterResponseModel    

logging.basicConfig(level=logging.DEBUG,
                   #filename='basic_config_test1.log',
                   format='%(asctime)s %(name)s %(levelname)s:%(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


app = Flask(__name__)
model = None
df = None

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def load_data():    
    # load data
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('DisasterMessages', engine)

    # load model
    model = DisasterResponseModel()
    model = model.load_model("./models/disaster_response_model_sverb.joblib")    
    #model = joblib.load("../models/disaster_response_model.joblib")

    return model, df


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    logger.info('route index')
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    ser_cat_sums = df.select_dtypes(exclude='object').sum()
    ser_cat_sums.drop('id', inplace =True   )
    cateogry_labels = ser_cat_sums.index.values
    
    logger.info(f'genre_name: {len(genre_names)}')
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=cateogry_labels,
                    y=ser_cat_sums
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    logger.info(f'ids: {ids}')
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    global model 
    global df
    model, df = load_data()

    app.run(host='0.0.0.0', port=3002, debug=True)


if __name__ == '__main__':
    main()