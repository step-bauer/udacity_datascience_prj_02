# Disaster Response Project

## Table of Contents
1. [Overview](#overview)
2. [Getting Started](#getting-started)	
	1. [Installation](#installation)
	2. [Executing Program](#execution)
	3. [Additional Material](#material)
3. [Project Details](#project-details)
4. [License](#license)
5. [Acknowledgement](#acknowledgement)
6. [Screenshots](#screenshots)

## Overview
This Repo contains the code for the "Disaster Response Project" which is the second prject of the Udacity Datascience Nanodegree.

It implements a message classification app that assigns a message to categories based on the message content.
The Project consists of the 3 parts:

* ETL Pipeline - Data cleaning and preparation
* ML part - Training of a classifier
* Web App - Simple application that allows user to enter a message and then assigns it ot categoires


## Getting Started

### Installation

#### 1. clone git repository
``` 
git clone https://github.com/step-bauer/udacity_datascience_prj_02.git
```

#### 2. create new virtual environment
```
conda -create -n <env_name> python=3.9

conda activate <env_name>

conda config --add channels conda-forge

conda install ptvsd pandas scipy scikit-learn matplotlib sqlalchemy joblib nltk flask plotly
```

for windows an alternative method is to use the conda-env-spec.txt
```
conda create -n py39_dresp --file conda-env-spec.txt
```

### Execution
change to main directory of the repository

##### 3.1 ETL Pipeline

```
python main.py etl-pipeline -mf ./data/disaster_messages.csv -cf ./data/disaster_categories.csv -db ./data/DisasterMsg.db
```

you can also call the process_pipline.py script directly and provide the three parameters (messages_filename:str, categories_filename:str, dbname:str)

##### 3.2 ML - Train Classifier

```
python main.py ml-pipeline -mf ./models/classifier_model.joblib -db ./data/DisasterMsg.db -tbl DisasterMessages
```

or as an alternative call train_classifier
```
python train_classifier.py ./data/DisasterMsg.db ./models/classifier_model.joblib
```

## Project Details

### Project structure
The following outlines the main elements of the project structure

- main dir
    - data
        - disaster_messages.csv
        - disaster_categories.csv
    - models
    - prep
    - web
        - templates
    - main.py
    - process_data.py
    - train_classifier.py

* data

The Project consists of the following parts:

### ETL Pipeline
The ETL pipeline will

* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

The code is stored in process_data.py

### ML Pipeline


### Flask Web App


