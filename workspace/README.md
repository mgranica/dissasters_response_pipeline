# Disaster Response Pipeline Project

## Dataset

> In order to classify the messages linked to natural disasters in order to focus the service on the people you need the most, we developed this project. The information comes from a database provided by figure eight. which has the content of the messages and registration of the category in which it is located.

you can find my repository here: https://github.com/mgranica/dissasters_response_pipeline

 
## Pipelines 

- **ETL pipeline:**   

    - Loads the messages and categories datasets
    - Merges the two datasets
    - Cleans the data
    - Stores it in a SQLite database  
        
- **ML pipeline:** 

To classify the messages we use the nltk libraries to normalize, tokenize, stop words removals, tagg parts of speach and lemmatize each of them. we fit the model with those inputs. to classify we use random forest and the decision tree of the scikit library.
    
   -  Loads data from the SQLite database
   - Splits the dataset into training and test sets
   - Builds a text processing and machine learning pipeline
   - Trains and tunes a model using GridSearchCV
   - Outputs results on the test set
   - Exports the final model as a pickle file
    

## web app

The project includes a web app so that in emergency situations, the classification of the message can be dynamically identified, communicated to the relevant authority and assistance to those in need as quickly as possible. 


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
