# Disaster_response_pipeline

# Table of Contents
1. [Project Overview](#project-overview)
2. [Required Libraries](#required-libraries)
3. [Project Structure](#project-structure)
4. [Running the Project](#running-the-project)
5. [Model Performance & Improvements](#model-performance--improvements)

## Project Overview
This project aims to classify messages sent during disasters using Natural Language Processing (NLP). The model categorizes messages into 36 different categories, including:

- **General Categories:** Genre, Related, Request, Offer
- **Aid & Medical Needs:** Aid-Related, Medical Help, Medical Products
- **Rescue & Security:** Search and Rescue, Security, Military
- **Basic Needs:** Water, Food, Shelter, Clothing, Money
- **Missing & Displaced Persons:** Missing People, Refugees
- **Casualties & Assistance:** Death, Other Aid
- **Infrastructure & Logistics:** Infrastructure-Related, Transport, Buildings, Electricity, Tools, Hospitals, Shops, Aid Centers, Other Infrastructure
- **Weather & Natural Disasters:** Weather-Related, Floods, Storm, Fire, Earthquake, Cold, Other Weather
- **Direct Reports:** Direct Report

Classifying these messages helps first responders and emergency teams prepare more effectively. For instance, if multiple messages from a flooded area indicate injuries and casualties, emergency responders can prioritize medical assistance. Additionally, these classifications can improve public awareness by providing real-time insights into disaster situations.

The project follows a structured approach:
1. Data cleaning and storage in a database.
2. Training a classifier using Grid Search for optimal hyperparameters.
3. Saving the trained model.
4. Deploying a web application to classify new messages and visualize data insights.

## Required Libraries
To run this project, the following packages and libraries are required:

- **SQLAlchemy**: Querying and saving data into the database.
- **pandas**: Data manipulation and preprocessing.
- **nltk**: Tokenization and text processing.
- **scikit-learn (sklearn)**: Machine learning framework for training and optimizing models.
- **xgboost**: Classification model for message categorization.
- **pickle**: Saving and loading the trained model.
- **Flask**: Web application framework.
- **Plotly**: Interactive data visualization.
- **Custom Modules**: `data_wrangling`, `data_clean`, and `data_saver` (included in the project files).

## Project Structure

### Root Directory:
- **App/**
  - `templates/`
    - `go.html`: Displays model predictions.
    - `master.html`: Main web app page with data visualizations.
  - `app.py`: Flask application for the web interface.

- **Data/**
  - `categories.csv`: Raw category data.
  - `messages.csv`: Raw messages data.
  - `data_clean.py`: Cleans duplicate and irrelevant data.
  - `data_wrangling.py`: Processes and transforms data.
  - `Disaster.db`: SQLite database for storing cleaned data.
  - `process_data.py`: Executes data processing and storage.

- **Models/**
  - `data_saver.py`: Module for saving data into the database.
  - `new_best_model.pkl`: Trained model saved using Pickle.
  - `train_classifier.py`: Script to train and save the classifier model.

## Running the Project
To run the project, follow these steps:

1. **Run the ETL pipeline:** This cleans and saves data into the database.
   ```sh
   python3 data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
   ```
2. Run the command to run the classifier trainer and save the model. This will also save the classifier metrics.
```sh
python3 models/train_classifier.py data/DisasterResponse.DB models/best_model.plk
```
3. lastly run to view the web app and start classifying messages. The terminal will display ip where you can click and the web app will be shown in the default browser.
```sh
python3 app/app.py 
```

## Model Performance and Improvments

The trained model achieved an overall accuracy of 0.95 on the test set. However, due to imbalanced training data, this metric may not fully reflect real-world performance. The recall score of 0.55 suggests moderate performance in identifying true positives.

### Potential Improvements:

 - Handling Imbalanced Data: Applying Synthetic Minority Over-sampling Technique (SMOTE) to increase instances of underrepresented categories or downsampling dominant cases.

 - Feature Engineering: Exploring advanced NLP techniques such as TF-IDF or word embeddings to improve text classification.

 - Hyperparameter Tuning: Further optimizing model parameters to enhance classification accuracy.

By addressing these aspects, the model can improve its reliability and effectiveness in real-world disaster response scenarios.
