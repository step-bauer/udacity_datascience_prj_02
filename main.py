import argparse
import ptvsd
import data.process_data 
import models.train_classifier
from  web.run import main as webmain

def _init_parser():
    """
    inits the argument parser for supporting command line invocation
    """
    parent_parser = argparse.ArgumentParser(description='Disaster Response Model CLI', add_help=True,
                                            fromfile_prefix_chars='@')

    parent_parser.add_argument ('-d', '--debug', dest='debug', action='store_true', 
                                help='activate debugging. attaches a debuggger on localhost and port 5679')

    parser_tasks = parent_parser.add_subparsers(prog='main.py', help="use sup coomands to specify a task. Use -h to see help",
                                                title='tasks', required=True,                                                
                                                dest='actionCmd')

    parser_etl_pipeline = parser_tasks.add_parser('etl-pipeline', #parents=[parent_parser],
                                                add_help=True, description='execute ETL Pipeline process')
    
    parser_etl_pipeline.add_argument('-mf', '--message-filename', required=True,
                        metavar=' MSG_FILENAME',
                        dest='messages_filename',
                        help='fullpath of the file that contains the messages')

    parser_etl_pipeline.add_argument('-cf', '--categories-filename', required=True,
                        metavar='CAT_FILENAME',
                        dest='categories_filename',
                        help='fullpath of the file that contains the categories')

    parser_etl_pipeline.add_argument('-db', '--database-name',
                        metavar='DBNAME',
                        dest='dbname',
                        help='name of the database with path. e.g. ./data/results.db')

    parser_etl_pipeline.add_argument('-tbl', '--tablename',
                        metavar='TBLNAME',
                        dest='tablename',
                        default='DisasterMessages',
                        help='name of the table the input data is stored at, e.g. DisasterMessages')


    parser_etl_pipeline.add_argument('-p', '--print-report',
                        dest='print_report',
                        action='store_true',
                        help='print ETL process report')


    #############
    # ML Parser
    # 
    parser_ml_pipeline = parser_tasks.add_parser('ml-pipeline', #parents=[parent_parser],
                                                add_help=True, description='execute ML Pipeline process')
    
    parser_ml_pipeline.add_argument('-mf', '--model-filename', required=True,
                        metavar=' MODLE_FILENAME',
                        dest='model_filename', 
                        help='fullpath of the file that the ML model is stored in')

    parser_ml_pipeline.add_argument('-db', '--database-name', required=True,
                        metavar='DBNAME',
                        dest='dbname',
                        default='./data/DisasterResponse.db',
                        help='name of the database with path. e.g. ./data/DisasterResponse.db')

    parser_ml_pipeline.add_argument('-tbl', '--tablename', required=True,
                        metavar='TBLNAME',
                        dest='tablename',
                        default='DisasterMessages',
                        help='name of the table the input data is stored at, e.g. DisasterMessages')

    parser_ml_pipeline.add_argument('-p', '--print-report',                         
                        dest='print_report',
                        action='store_true',
                        help='print ML process report')                        


    ###################################################################################################
    #
    #  run web app
    #
    parser_web_app = parser_tasks.add_parser('run-web-app', #parents=[parent_parser],
                                                add_help=True, description='execute web app')

    parser_web_app.add_argument('-db', '--database-name', required=True,
                        metavar='DBNAME',
                        dest='dbname',
                        default='./data/DisasterResponse.db',
                        help='name of the database with path. e.g. ./data/DisasterResponse.db')

    parser_web_app.add_argument('-tbl', '--tablename', required=True,
                        metavar='TBLNAME',
                        dest='tablename',
                        default='DisasterMessages',
                        help='name of the table the input data is stored at, e.g. DisasterMessages')

    parser_web_app.add_argument('-mf', '--model-filename', required=True,
                        metavar=' MODLE_FILENAME',
                        dest='model_filename', default='DisasterMessages',
                        help='fullpath of the file that the ML model is stored in')
                        
    parser_web_app.add_argument('-p', '--port', required=True,
                        metavar='PORT',
                        dest='port',
                        default='3002',
                        help='port that the web app runs on')


    return parent_parser
                            

# main method for invoking via command line
if __name__ =='__main__':
    parser = _init_parser()

    args = parser.parse_args()

    if args.debug:
        ptvsd.enable_attach(address=('localhost',5679))
        print('waiting for debugger on localhost and port 5679....')
        ptvsd.wait_for_attach()

    if args.actionCmd == 'etl-pipeline':
        data.process_data.run_etl_pipeline(args.messages_filename, 
                                           args.categories_filename, 
                                           args.dbname, 
                                           args.tablename, 
                                           args.print_report)
    elif args.actionCmd == 'ml-pipeline':
        models.train_classifier.main(args.dbname, args.tablename, args.model_filename)
    elif args.actionCmd == 'run-web-app':
        webmain(args.dbname, args.tablename, args.model_filename, args.port)