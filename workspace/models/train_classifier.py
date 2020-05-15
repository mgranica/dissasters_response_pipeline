import sys
import pandas as pd
import numpy as np
import re

from sqlalchemy import create_engine
import sqlite3

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

import pickle

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    conn = sqlite3.connect('DisasterResponse.db')    
    df= pd.read_sql('SELECT * FROM messages;', conn)
    df.head()
    X = df['message'].values
    y = df.loc[:, 'related':'direct_report'].values
    category_names = list(df.columns[4:])
    
    return X, y, category_names
    #pass

   


def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
        
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens



def build_model():
    pipeline = pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidt', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    max_depths = [1, 2]
    min_samples_leaf = [5,10]
    min_samples_split = [5,10]

    parameters = {
        'clf__estimator__max_depth': max_depths,
        'clf__estimator__min_samples_leaf': min_samples_leaf,
        'clf__estimator__min_samples_split': min_samples_split
    }

    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        # print out each category name and its corresponding classification report
        print('Category: {}'.format(category_names[i]))
        print(classification_report(Y_test[:, i], y_pred[:, i]))
        print('\n')  
        

        

def save_model(model, model_filepath):
    
    pickle_out = open(model_filepath, 'wb')
    pickle.dump(model, pickle_out)
    pickle_out.close()
    pass


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