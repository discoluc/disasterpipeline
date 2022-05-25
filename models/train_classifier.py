# import libraries
import sys
from matplotlib.pyplot import grid
import pandas as pd
import re
from sqlalchemy import create_engine
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score


# nltk installation/download
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")


def load_data(database_filepath):
    """
    Loads the data from a sql db and splits the db in a dependent and independent variable.
    Furthermore it outputs the categories of the classification
    """
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("messages", engine)

    # Splitting into the dependend variable and the categories
    X = df["message"].astype(str)
    y = df.drop(["id", "original", "message", "genre"], axis=1)
    cat = y.columns
    return X, y, cat


def tokenize(text):
    """
    Input: a string (several sentences)
    Removes all punctuation and sets everything to lowercase. The text is tokenized 
    into singular words which are lemmatized.
    """
    # Removing punctuation and setting to lowercase
    text = re.sub(r"[^a-z0-9]", " ", text.lower())
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w.strip()) for w in words]
    return lemmed


def build_model(
    classifier=RandomForestClassifier(), grid_search=False, grid_search_parameters={}
):
    """
    Model setup. One can select of grid search is used to optimize the model or if one uses the
    standard model/pipeline. The pipeline consists of a CountVectorizer, a TfidfTransformer and
    a Multioutputclassifier. Here one can select the estimator. Which is in the standard setup the Random forest classifier.
    
    """
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(classifier, n_jobs=-1)),
        ]
    )
    if grid_search:
        model = GridSearchCV(pipeline, param_grid=grid_search_parameters)
    else:
        model = pipeline

    return model


def evaluate_model(model, X_test, y_test, category_names):
    """
    Here the scores for each class gets posted including the accuracy score.
    """
    # Prediction on the test set
    y_pred = model.predict(X_test)

    # looping over all categorys and printing the report
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(y_test.iloc[:, i], y_pred[:, i]))
        print(
            "accuracy: " + str("%.2f" % accuracy_score(y_test.iloc[:, i], y_pred[:, i]))
        )


def save_model(model, model_filepath):
    """
    Saves the model into a pickle file.
    
    """
    with open(model_filepath, "wb") as pckfiles:
        pickle.dump(model, pckfiles, pickle.HIGHEST_PROTOCOL)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        parameters = {"clf__estimator__max_features": ["sqrt", "log2"]}
        print("Building model...")

        model = build_model(grid_search=True, grid_search_parameters=parameters)

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
