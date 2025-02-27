import sys
import pandas as pd
from sqlalchemy import create_engine

#remove Duplicates fuction
def remove_duplicates(df):
   """
   Remove duplicate rows and rows where 'related' column equals 2.
   """
   df.drop_duplicates(inplace=True)
   df = df[df['related'] != 2]
   return df

# split catagores and create columns for each catagory 3 and 4
def series_str_split(column):
    """
    Split the 'categories' column into separate category columns.
    
    Args:
    column: pandas Series. Series containing the categories in string format.
    
    Returns:
    pd.DataFrame. DataFrame with each category as a separate column.
    """
    value_dict = {}
    for i in range(len(column)):
        the_list = column.iloc[i].split(';')
        for j in the_list:
            key, value = j.split('-')
            if key in value_dict:
                value_dict[key] += [int(value)]
            else:
                value_dict[key] = [int(value)]
    return pd.DataFrame(value_dict)

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.
    
    Args:
    messages_filepath: str. Filepath for the messages dataset.
    categories_filepath: str. Filepath for the categories dataset.
    
    Returns:
    df: pd.DataFrame. DataFrame obtained by merging messages and categories datasets.
    """
   
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(categories , messages, on='id')

     # Split categories into separate columns
    categories_split = series_str_split(df['categories'])
    df.drop(columns=['categories'], inplace=True)
    
    return pd.concat([df, categories_split], axis=1)


def clean_data(df):
    '''
    Calls function remove duplicate rows.
    '''
    return remove_duplicates(df)


def save_data(df, database_filename):
    '''
    Saves df into the initiated database under message table
    '''
    db_path = 'sqlite:///'+database_filename
    engine = create_engine(db_path)
    df.to_sql('Message', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
