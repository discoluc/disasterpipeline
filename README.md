# Disaster Response Pipeline Project

### Table of Contents
1. [Summary](#summary)
2. [Installation](#installation)
3. [Instructions](#instruction)
4. [Files](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)


## Summary <a name="summary"></a>
In this project I build a web app which can classify messages into different classes e.g. 'needs_water' or 'needs_food'. This is very useful in the case of a desaster where one needs to classify incoming messages fast. The data is prepared and cleaned in an etl script in data. And in the model folder the machine learning model is built which is used to classify the messages.

## Installation <a name="installation"></a>

The code only uses Pandas, Numpy, scikit, flask, nltk, sqlalchemy and was tested on a Python version 3.8.8.
There should be no necessary  other libraries to run the code here beyond these.


## Instructions <a name="instruction"></a>:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Files <a name=files></a>
The two csv's disaster_messages and disaster_catepories represent both input files with the original tweet and to which category they belong to. This datasets are combined and clenead and put into the Database DisasterResponse. This process is in the process_data.py.
Finally the database is used to train the model with the train_classifier py. The resulting model is saved to the classifier.pkl file.

The flask web app is contained in the app folder. Where the run.py file is located which is used to run the web app.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
This web app was developed during an exercise regarding the [Udacity Data Science Nanodegree](https://www.udacity.com/school-of-data-science), feel free to use the code as you like.

The data which is used in this project was provided by [Figure Eight (now Appen)](https://appen.com/).



