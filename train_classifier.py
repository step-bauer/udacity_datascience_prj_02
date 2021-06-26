"""
This module provides functionality to train the Disaster Response message classifier absed on the 
data the ETL process has stored in the Disaster Response database.

This module can also be executed as script from command line or be invoked via the 
main.py command line argument script which provies some help how to call it.
"""
import logging
import re
import sys

from joblib import dump, load
import pandas as pd
from sqlalchemy import create_engine

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download(['punkt','stopwords','wordnet','averaged_perceptron_tagger'])


logging.basicConfig(level=logging.DEBUG,
                   #filename='basic_config_test1.log',
                   format='%(asctime)s %(name)s %(levelname)s:%(message)s')

logger = logging.getLogger(__name__)

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    ''' Determines if first word
        in a body of text is a verb
    '''
    def starting_verb(self, text:str)->bool:
        """
        checks if text begins with a verb

        Parameters
        -----------
            text :str
                text is a sentence that is to be checked

        Returns
        -------
            bool  : True if text begins with a verb
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(DisasterResponseModel.tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        """
        fit method - does nothinng
        """
        return self

    def transform(self, X):
        """
        transform method - applies on each element of pandas.Series X the method starting_verb 
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


class DisasterResponseModel ():
    """
    This class encaspulates the Disaster Response Model. It provides functionality to load message and hot encoded category data from
    a sqlite database that has been preprocessed by the Disaster Response ETL pipeline.

    It also provides functionality to build a ML pipeline and GridSearchCV model and to split, train and test the model. And finally to
    predict classification and to determine the model quality.
    """
    
    # regular expression pattern to find URL
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # compiled regular expression to find URL in strings
    url_reg = re.compile(url_pattern)

    lemmatizer = WordNetLemmatizer()
        
    def __init__(self):
        """
        class init
        """
        self.df = None
        # features and labels
        self.X, self.Y = None, None
        # training and test data sets
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        # predictions
        self.y_pred = None

        self.pipeline = self.build_pipeline()
        
        self.grid_parameters = {
                                'tfidf_vect__ngram_range': ((1, 1), (1, 2)),
                                'tfidf_vect__max_df': (0.5, 0.75, 1.0),
                                'tfidf_vect__max_features': (None, 5000, 10000),
                                #'tfidf_vect__tfidf__use_idf': (True, False),
                                #'tfidf_vect__stop_words': (None, stopwords.words('english'))
                            }

        self.model = None


    def load_data (self, db_filename:str, table_name:str)->pd.DataFrame:
        """loads data from a sqllite database into a pandas dataframe

        Parameters
        ----------
            db_filename:str
                filename of the DB file that the data is to be loaded from, e.g. disasterResponse.db
            
            table_name:str
                name of the table that the data is read from (SELECT * from <table_name>), e.g. DisasterResponse

        Returns
        -------

            pandas.DataFrame : loaded dataset
        """
        # load data from database
        logger.debug('db_filename %s / table_name %s', db_filename, table_name)
        db_engine_load_str = f'sqlite:///{db_filename}'
        logger.debug('db_engine_load_str %s', db_engine_load_str)

        engine = create_engine(db_engine_load_str)

        try:
            df = pd.read_sql(f'SELECT * from {table_name}', engine)
            logger.debug('Dataframe successfully loaded. Shape: %s', df.shape[0])
        except Exception as exception:        
            logger.error('Table %s does not exist', table_name)        

            # list available tables
            print('\nAvailable Tables\n'+'-'*30+'\n')
            sql_statement = "SELECT name FROM sqlite_master WHERE type ='table' AND name NOT LIKE 'sqlite_%'"
            with engine.connect() as con:
                rs = con.execute(sql_statement)
                for rec in rs:                
                    print(rec)

            print('\n'+'-'*30+'\n')

            raise exception

        self.df = df
        logger.debug('DataFrame columns: %s',df.columns)
        self.X, self.Y = self._split_data(df)

        return df

    def _split_data(self, df):
        """split data set in labels and features
        """
        X = df['message']
        Y = df.drop(columns=['message','original','genre','id'])    

        logger.debug('Feature columns: %s',X.shape)
        logger.debug('Label columns: %s', Y.columns)

        return X, Y

    @classmethod
    def tokenize(cls, text:str, enable_lemmatizer:bool=False)->list:
        """
        this tokenizer method splits text in word tokens. 

        Parameters
        ----------
            text:str
                text to be tokenized
            
            enable_lemmatizer:bool
                if true then lemmatizing is activated

        Returns
        -------

            list : list of tokens extracted from text input

        """
        # replace url
        text = cls.url_reg.sub('urlplaceholder',text)
        # tokenize text
        tokens = word_tokenize(text)
        
        # lemmatize andremove stop words
        if enable_lemmatizer:
            tokens = [cls.lemmatizer.lemmatize(word.lower().strip()) for word in tokens]
        else:
            tokens = [word.lower().strip() for word in tokens]

        return tokens


    #sklearn.pipeline.Pipeline
    def build_pipeline (self, starting_verb:bool=False)->Pipeline:
        """builds a sklearn.pipeline Pipeline

        Parameter
        ---------
            starting_verb:bool
                if true then additional feature starting verb is added to the pipeline process
        """
        if not starting_verb:
            self.pipeline = Pipeline([                
                ('tfidf_vect', TfidfVectorizer(tokenizer=self.tokenize)),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
            ])
        else:    
            self.pipeline = Pipeline([
                ('features', FeatureUnion([
                    ('text_pipeline', Pipeline([
                        ('vect', CountVectorizer(tokenizer=self.tokenize)),
                        ('tfidf', TfidfTransformer())
                    ])),

                    ('starting_verb', StartingVerbExtractor())
                ])),

                ('clf', MultiOutputClassifier(RandomForestClassifier()))
            ])

        return self.pipeline
        

    def display_test_results(self):
        """
        displays ther results of the model
        """
        target_names = self.y_test.columns.values
        print(metrics.classification_report(self.y_test.reset_index(drop=True), self.y_pred, target_names=target_names))
        
        

    def build_model(self, is_grid_search:bool=False):
        """
        Run the ML pipeline that splits tha data in train and test set, 
        creates the model by transforming and fitting the train data and 
        finally uses test data set to to test the prediciton quality

        Parameters
        ----------
            is_grid_search : bool
                if True grid search will be used otherwise just the default pipeline is used

        Returns
        -------

            metrics.classification_report : sklearn classification report 
        """                
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.01, train_size=0.01)
        logger.debug('split data set into %s train records and %s test records', self.X_train.shape[0], self.X_test.shape[0])
        
        if not is_grid_search:
            logger.debug('fit simple pipline')
            #run the pipeline
            self.pipeline.fit(self.X_train, self.y_train)

            logger.debug('execute prediction for X_test data set')
            self.y_pred = self.pipeline.predict(self.X_test)
        elif is_grid_search:            
            self.model = GridSearchCV(self.pipeline, param_grid = self.grid_parameters)            

            logger.debug('fit GridSearch on train data set')
            self.model.fit(self.X_train, self.y_train)

            logger.debug('execute prediction for X_test data set')
            self.y_pred = self.model.predict(self.X_test)
        
 
    def save_model (self, filename:str):
        """
        save the model to a pickle file so that the rained model can be loaded again without training

        Parameters
        ----------

                filename : str
                    name of the file the model is saved to
        """
        # uses dump from joblib
        dump(self, filename)
        dump(self.model, f'model_{filename}')        
        logger.debug('model saved to file %s', filename)

    @classmethod
    def load_model (cls, filename:str):
        """
        loads a model from a pickle file

        Parameters
        ----------

                filename : str
                    name of the file the model is loaded from
        """
        disaster_response_model = load(filename=filename)

        return disaster_response_model


def main (db_filename:str, table_name:str, model_out_filename:str):
    """
    Parameters
    ----------
        db_filename:str
            filename of the DB file that the data is to be loaded from, e.g. disasterResponse.db
        
        table_name:str
            name of the table that the data is read from (SELECT * from <table_name>), e.g. DisasterResponse
    """
    disaster_response_model  = DisasterResponseModel()
    disaster_response_model.load_data (db_filename, table_name)

    disaster_response_model.build_model(False)
    disaster_response_model.save_model(model_out_filename)
    disaster_response_model.display_test_results()
    
    

# main method for invoking via command line
# use main.py instead of this script to get help how to invoke it
if __name__ =='__main__':
    # db_filename:str, table_name:str, model_out_filename:str
    main(sys.argv[0], sys.argv[1], sys.argv[2])
    
    
