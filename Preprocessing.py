import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize

# Download NLTK resources for sentence tokenization
nltk.download('punkt')

# Read in the training dataset
def main(df):
    # Read the CSV file, specifying a quote character
    try:
        unprocessed_df = pd.read_csv(df, quotechar='"')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Preprocess the conversations to fix missing commas
    unprocessed_df = fix_missing_commas(unprocessed_df)
    
    # Lowercase the text
    lowercased_df = lowercasing(unprocessed_df)
    
    # Add correct speaker labels
    labeled_df = add_labels(lowercased_df)
    
    # Tokenize the conversation by sentence
    tokenized_df = tokenize_by_sentences(labeled_df)
    
    # Print the resulting dataframe
    print(tokenized_df)
    
# Fix missing commas between dictionary entries in a list
def fix_missing_commas(dataframe):
    # Define the regex pattern to detect a missing comma between dictionaries
    pattern = r"(\}\s*\{)"
    
    # Apply the fix by adding a comma between the dictionaries where needed
    dataframe["conversations"] = dataframe["conversations"].apply(lambda x: re.sub(pattern, "}, {", str(x)))
    
    return dataframe

# Lowercase the text
def lowercasing(dataframe):
    dataframe["conversations"] = dataframe["conversations"].str.lower()
    return dataframe
    
# Add correct speaker labels
def add_labels(l_df):
    human_label = "'from': 'human', 'value':"
    gpt_label = "'from': 'gpt', 'value':"
    
    # Replace occurrences of the labels in the "conversations" column directly
    l_df["conversations"] = l_df["conversations"].replace(human_label, "[COUNSELEE]", regex=True)
    l_df["conversations"] = l_df["conversations"].replace(gpt_label, "[THERAPIST]", regex=True)
    
    return l_df

# Tokenize the conversation by sentence
def tokenize_by_sentences(dataframe):
    dataframe["tokenized_sentences"] = dataframe["conversations"].apply(sent_tokenize)
    return dataframe

if __name__ == "__main__":
    df = input("Insert Dataset Name: ")
    main(df)
