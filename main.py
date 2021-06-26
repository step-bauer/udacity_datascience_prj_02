import argparse
import ptvsd
import process_data
import train_classifier

def _init_parser():
    """
    inits the argument parser for supporting command line invocation
    """
    parent_parser = argparse.ArgumentParser(description='Disaster Response Model CLI', fromfile_prefix_chars='@')

    parent_parser.add_argument ('-d', '--debug', dest='debug', action='store_true', 
                                    help='activate debugging. attaches a debuggger on localhost and port 5679')

    parser_tasks = parent_parser.add_subparsers(prog='main.py', 
                                            title='tasks', 
                                            dest='actionCmd')

    parser_etl_pipeline = parser_tasks.add_parser('etl-pipeline', #parents=[parent_parser],
                                                add_help=False, description='execute ETL Pipeline process')
    
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

    parser_etl_pipeline.add_argument('-p', '--print-report',                         
                        dest='print_report',
                        action='store_true',
                        help='print ETL process report')


    #############
    # ML Parser
    # 
    parser_ml_pipeline = parser_tasks.add_parser('ml-pipeline', #parents=[parent_parser],
                                                add_help=False, description='execute ML Pipeline process')
    
    parser_ml_pipeline.add_argument('-mf', '--model-filename', required=True,
                        metavar=' MODLE_FILENAME',
                        dest='model_filename',
                        help='fullpath of the file that the ML model is stored in')

    parser_ml_pipeline.add_argument('-db', '--database-name', required=True,
                        metavar='DBNAME',
                        dest='dbname',
                        help='name of the database with path. e.g. ./data/DisasterResponse.db')

    parser_ml_pipeline.add_argument('-tbl', '--tablename', required=True,
                        metavar='TBLNAME',
                        dest='tablename',
                        help='name of the table the input data is stored at, e.g. DisasterMessages')

    parser_ml_pipeline.add_argument('-p', '--print-report',                         
                        dest='print_report',
                        action='store_true',
                        help='print ML process report')                        

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
        process_data.run_etl_pipeline(args.messages_filename, args.categories_filename, args.dbname, args.print_report)
    elif args.actionCmd == 'ml-pipeline':
        train_classifier.main(args.dbname, args.tablename, args.model_filename)