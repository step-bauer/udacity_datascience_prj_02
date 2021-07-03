""" The process_data module contains the code for the ETL Pipeline of the
    Disaster Response Project.

    Main tasks are to
    * Load the messages and categories datasets
    * Merge the two datasets
    * Clean the data
    * Store the results in a SQLite database

"""
from __future__ import absolute_import

import logging
import sys

import pandas as pd
import sqlalchemy as sql
import os

log_level = os.getenv('LOGLEVEL', logging.ERROR)
logging.basicConfig(level=log_level,
                   format='%(asctime)s %(name)s %(levelname)s:%(message)s')

logger = logging.getLogger(__name__)

class ETLPipeline ():
    """ETLPipeline class provides methods to control the ETL process.
    """

    def __init__(self, messages_filename:str, categories_filename:str, dbname : str):
        """inits ETLPipeline

        Parameters
        ----------

                messages_filename:str
                    name of the message data file. Full path.
                
                categories_filename:str
                    name of the categories data file. Full path.
                
                dbname : str
                    name of the database with path. e.g. ./data/results.db
        """
        self.messages_filename      = messages_filename
        self.categories_filename    = categories_filename
        self.dbname                 = dbname

        self.df_messages    = None
        self.df_categories  = None
        self.df_merged      = None
        self.process_report = []

    
    def load_data(self):
        """load data sets from datafiles. """

        def load_data_from_file (filename):
            logger.debug({'logmessage':'try to load data file: {}'.format(filename)})
            df = pd.read_csv(filename)
            logger.info({'logmessage':f'loaded {df.shape[0]} records from {filename}'})    

            return df

        self.df_messages    = load_data_from_file(self.messages_filename)
        self.df_categories  = load_data_from_file(self.categories_filename)

        self.process_report.append({'stepname' : "load_data",
                                   'messages' :  [ f'loaded {self.df_messages.shape} records from {self.messages_filename}',
                                                f'loaded {self.df_categories.shape} records from {self.categories_filename}']})


    def merge_data(self):
        """executes a left join of messages and categories dataframe based on id column.
        """
        self.df_merged = pd.merge(self.df_messages, self.df_categories, on='id', how='left')
        logger.info({'logmessage':f'merged dataframes of messages with categories. Shape is {self.df_merged.shape}'})

        self.process_report.append({'stepname' : "merge_data",
                                   'messages' :  [ f'merged dataframes of messages with categories. Shape is {self.df_merged.shape}']})


    def _get_seperated_categories(self):
        """ Split the values in the categories column on the ";" character so that 
            each value becomes a separate column. 

            Categories column contains valuesl like "buildings-0;electricity-0;tools-1". This
            will be split into 3 columns buildings, electricity, tools.

            In addition we convert the string values to One-Hot-Encoding

            Return
            -------

                pandas.DataFrame() : dataframe object with separated categories as columns
        """
        category_headers = self.df_merged['categories'][0]
        category_headers = category_headers.replace('-0','').replace('-1','')

        category_headers = category_headers.split(';')
  
        # the next line removes all alpha caracters and just keeps 0, 1 and ;
        # after splitting by ";" we have the one-hot encoding for the categories
        categories_sep = self.df_merged['categories'].str.replace('[^01;]','', regex=True).str.split(';',expand=True)
        categories_sep.columns = category_headers

        # add id column
        categories_sep['id'] = self.df_merged['id']

        # convert strings to int values
        for col in  categories_sep.columns:
            if col != 'id':
                try:
                    categories_sep[col] = categories_sep[col].astype('int64')
                except Exception as ex:
                    logger.error(f'category column could not be converted to int64: Column {col} failed.')
                    raise ex
    
        return categories_sep
    
    
    def clean_data(self):
        """this cleans the merged dataframe data

        """

        #check if data has been loaded
        if self.df_merged is None:
            logger.error("execute merge() before you call clean_data. Process will be stopped.")
            return

        # there are some categories "related-2" - we rename them to related-1 as the value should be
        # 0 for not set or 1 for set - 2 makes no sense and seems to be wrong
        self.df_merged['categories'] = self.df_merged['categories'].str.replace('related-2','related-1')

        df_seperated_categories = self._get_seperated_categories()

        # create new dataframe with separated categories
        self.df_merged = pd.merge(self.df_merged, df_seperated_categories, on = 'id', how='inner')

        #remove duplicates
        self.df_merged.drop_duplicates('id', inplace=True)        
        self.df_merged.drop(labels=self.df_merged[self.df_merged['message']=='#NAME?'].index, axis=0, inplace=True)

        # drop category column
        self.df_merged.drop(columns='categories', inplace=True)

        self.process_report.append({'stepname' : "clean_data",
                                   'messages' :  [ f'cleaned merged dataframe. Shape is {self.df_merged.shape}']})


    def run_etl_pipeline (self):
        """runs the complete pipeline which includes these step in this order

            * load_data
            * merge_data
            * clean_data
            * store_data
        """
        self.load_data()
        self.merge_data()
        self.clean_data()
        self.store_data()
        

    def store_data(self):
        """store the data in a sqllite database
        
        """
        db_name = f'sqlite:///{self.dbname}'
        engine = sql.create_engine(db_name)
        tbl_name = 'DisasterMessages'
        self.df_merged.to_sql(tbl_name, engine, index=False, if_exists='replace')

        msg = f'stored merged dataframe in sqlite DB: {self.dbname} - Tablename: {tbl_name}'
        logger.info(msg)
        self.process_report.append({'stepname' : "store_data",
                                    'messages' :  [msg] })


    def print_report(self):
        """print a small ETL process report
        """
        print()
        print('-'*15+'   Report   '+'-'*15)
        print()
        for step in self.process_report:
            print(step['stepname'])

            for msg in step['messages']:
                print(msg)

            print('-'*30)
            print()
    #
    #   Properites
    #
    @property
    def dbname(self):
        """property getter for dbname"""
        return self._dbname

    @dbname.setter
    def dbname(self, value):
        """property setter for dbname"""
        self._dbname = value

    @property
    def process_report(self):
        """property getter for process_report"""
        return self._process_report

    @process_report.setter
    def process_report(self, value):
        """property setter for process_report"""
        self._process_report = value



    @property
    def messages_filename(self):
        """property getter for messages_filename"""
        return self._messages_filename

    @messages_filename.setter
    def messages_filename(self, value):
        """property setter for messages_filename"""
        self._messages_filename = value

    @property
    def categories_filename(self):
        """property getter for categories_filename"""
        return self._categories_filename

    @categories_filename.setter
    def categories_filename(self, value):
        """property setter for categories_filename"""
        self._categories_filename = value


    @property
    def df_messages(self):
        """property getter for df_messages"""
        return self._df_messages

    @df_messages.setter
    def df_messages(self, value):
        """property setter for df_messages"""
        self._df_messages = value

    @property
    def df_categories(self):
        """property getter for df_categories"""
        return self._df_categories

    @df_categories.setter
    def df_categories(self, value):
        """property setter for df_categories"""
        self._df_categories = value

    @property
    def df_merged(self):
        """property getter for df_merged"""
        return self._df_merged

    @df_merged.setter
    def df_merged(self, value):
        """property setter for df_merged"""
        self._df_merged = value



def run_etl_pipeline (messages_filename:str, categories_filename:str, dbname:str, is_print_report:bool=False):
    """run the ETL pipeline process

    Parameters
    ----------

        messages_filename:str
            name of the file that contains the messages
        
        categories_filename:str
            name of the file that contains the categories

        dbname : str
            name of the database

        is_print_report:bool
            if True then a small ETL process report is printed
    """
    logger.debug({'logmessage':'run_etl_pipeline', 
                   'messages_filename':messages_filename, 
                   'categories_filename':categories_filename})

    etl_pipeline = ETLPipeline(messages_filename, categories_filename, dbname)
    etl_pipeline.run_etl_pipeline()

    if is_print_report:
        etl_pipeline.print_report()

# main method for invoking via command line
if __name__ =='__main__':
    # messages_filename:str, categories_filename:str, dbname:str
    run_etl_pipeline(sys.argv[0], sys.argv[1], sys.argv[2])

    
    
