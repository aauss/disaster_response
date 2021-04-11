# Disaster Response Pipeline Project

This repo contains my Udacity project on a disaster response pipeline. The idea in this project is to develop a end-to-end pipeline that consists of an ETL-pipeline that feeds a machine learning model. The model and the data visualizations is then served in a Flask web app.

### Installation

To run this app, you need to install the libraries listed in `env.yml`. If you have Anaconda/Miniconda, just run `conda env create -f env.yml` to install all necessary libraries. After succesfully installing the libraries, you need to activate the environment with `conda activate disaster`to be able to run the pipeline.

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves the model (NOTE: This takes some minutes.)
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ to see the web app.
