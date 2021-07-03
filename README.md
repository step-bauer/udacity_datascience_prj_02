# Disaster Response Project


## Installation

### Window Installation
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

#### 3. install requirements

## 2nd Project for Udacity Datascience Nanadegree
This Repo contains the code for the "Disaster REsponse Project" which is the second prject of the Udacity Datascience Nanodegree.


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


