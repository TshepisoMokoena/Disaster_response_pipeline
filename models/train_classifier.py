import sys
import re
import string
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report,accuracy_score
import pickle


def load_data(database_filepath):
    """
    Load data from the SQLite database and split into feature and target variables.
    
    Args:
    database_filepath: str. Filepath for the database.
    
    Returns:
    X: pd.DataFrame. Feature variables.
    Y: pd.DataFrame. Target variables.
    category_names: list of str. List of category names.
    """
    # Read data from database
    db_path = 'sqlite:///' + database_filepath
    engine = create_engine(db_path)
    df = pd.read_sql_table('Message', engine)
    
    # Create dummy variables for the genre column
    one_hot_encoded_genre = pd.get_dummies(df['genre'], prefix='genre')
    
    # Define feature and target variables
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize and clean text data.
    
    Args:
    text: str. Text data to be tokenized.
    
    Returns:
    clean_tokens: list of str. List of cleaned tokens.
    """
    text_no_pun = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    stop_words = set(stopwords.words('english'))  # Get stop words
    tokens = word_tokenize(text_no_pun)
    removed_stop = [word for word in tokens if word.lower() not in stop_words]  # Remove stop words
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in removed_stop]
    
    return clean_tokens


def tokenize(text):
    """
    Tokenize and clean text data.
    
    Args:
    text: str. Text data to be tokenized.
    
    Returns:
    clean_tokens: list of str. List of cleaned tokens.
    """
    text_no_pun = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    stop_words = set(stopwords.words('english'))  # Get stop words
    tokens = word_tokenize(text_no_pun)
    removed_stop = [word for word in tokens if word.lower() not in stop_words]  # Remove stop words
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in removed_stop]
    
    return clean_tokens


def build_model():
    """
    Build a model with GridSearchCV to find the best parameters.
    
    Returns:
    model: GridSearchCV. Model with the best parameters found by GridSearchCV.
    """
    # Define Pipeline
    pipeline_xgb = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(XGBClassifier()))
    ])

    # Define parameters for GridSearchCV
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__max_depth': [3, 5, 7],
        'clf__estimator__n_estimators': [50, 100, 200]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(pipeline_xgb, param_grid=parameters, cv=3, verbose=2)

    return grid_search


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model by calculating precision, recall, F1-score, and accuracy for each category.
    
    Args:
    model: sklearn model. Trained model.
    X_test: pd.DataFrame. Test features.
    Y_test: pd.DataFrame. True labels for test data.
    category_names: list of str. List of category names.
    
    Returns:
    transposed_df: pd.DataFrame. DataFrame containing evaluation metrics for each category.
    """
    # Predict on test data
    predictions = model.predict(X_test)
    
    # Initialize lists to store metrics
    precision_list = []
    recall_list = []
    f1_score_list = []
    accuracy_list = []

    # Loop through each category and compute metrics
    for idx, label in enumerate(category_names):
        y_true = Y_test.iloc[:, idx]
        y_pred = predictions[:, idx]
        
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)
        accuracy_list.append(accuracy)

    # Create a DataFrame with metrics for each category
    metrics_df = pd.DataFrame({
        'Label': category_names,
        'Precision': precision_list,
        'Recall': recall_list,
        'F1-score': f1_score_list,
        'Accuracy': accuracy_list
    })

    # Set the label as the index and transpose the DataFrame
    metrics_df.set_index('Label', inplace=True)
    transposed_df = metrics_df.transpose()
    
    return transposed_df


def save_model(model, model_filepath):
    """
    Save the model to a pickle file.
    
    Args:
    model: sklearn model. Trained model to be saved.
    model_filepath: str. Filepath to save the pickle file.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def save_data_to_db(df, table_name):
    '''  
    Save dataframe to SQLite database.
    
    Args:
    df: pd.DataFrame. Dataframe to be saved.
    table_name: str. Name of the table to save the dataframe.
    '''
    try:
        engine = create_engine('sqlite:///data/DisasterResponse.db')
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print('Dataframe saved to DB')
    except Exception as e:
        print('Failed to save data to table, error:', e)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=66)
        
        print('Building model...')
        model = build_model()
        
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        Model_metrics = evaluate_model(model, X_test, Y_test, category_names)
        
        #save metrics to DB
        save_data_to_db(Model_metrics, 'Model_Metrics')

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
