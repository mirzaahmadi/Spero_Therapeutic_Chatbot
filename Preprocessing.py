# Import necessary libraries for preprocessing
import pandas as pd

# Read in the training dataset
def main(df):
    unprocessed_df = pd.read_csv(df)
    preprocess(unprocessed_df)

# Go through the preprocessing steps: Lowercasing, Removing punctuation, Removing stopwords
def preprocess(dataframe):
    dataframe["conversations"] = dataframe["conversations"].str.lower()
    


if __name__ == "__main__":
    df = input("Insert Dataset Name: ")
    main(df)