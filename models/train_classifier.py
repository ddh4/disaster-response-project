import sys
import re
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import pickle


def load_data(database_filepath):

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM Messages", engine)

    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    category_names = Y.columns.values

    return X, Y, category_names


def tokenize(text):
    '''
    Override the string tokenization step while preserving the preprocessing and n-grams generation steps.
    '''
    # Normalize text
    normalized = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Word tokenize
    words = word_tokenize(normalized)

    # Remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]

    # Lemmatize
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]

    return lemmed


def build_model():

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfid', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1,1), (1,2), (1,3)),
        'clf__estimator__n_estimators': [5, 10]
    }

    cv = GridSearchCV(pipeline, param_grid = parameters, cv=3, verbose = 2)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):

    predictions = pd.DataFrame(model.predict(X_test), columns=category_names)

    for column in category_names:
        print("Report for {}:\nAccuracy: {:0.2f}% \n{}".format(column, accuracy_score(Y_test[column], predictions[column]), classification_report(Y_test[column], predictions[column])))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
