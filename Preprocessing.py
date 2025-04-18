""" 
This file will take in the original dataset (original_therapy_transcript) and preprocess it, returning a preprocessed dataset (which is housed in the 'data' directory)
"""

import pandas as pd
import nltk
import re

nltk.download('punkt') # Download the NLTK resources for sentence tokenization

# Read in the training dataset
def main(df):
    # Read the CSV file, specifying a quote character
    try:
        unprocessed_df = pd.read_csv(df, quotechar='"')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    lowercased_df = lowercasing(unprocessed_df) # Lowercase the text
    labeled_df = add_labels(lowercased_df) # Add correct speaker labels
    sans_whitespace_df = remove_whitespace(labeled_df) # Remove any whitespace
    commas_fixed_df = fix_missing_commas(sans_whitespace_df) # Preprocess the conversations to fix missing commas
    
    # Save the resulting preprocessed dataframe
    print(commas_fixed_df)
    commas_fixed_df.to_csv("preprocessed_dataset.csv")

# Lowercase the text
def lowercasing(dataframe):
    dataframe["conversations"] = dataframe["conversations"].str.lower()
    return dataframe
    
# Add and format speaker labels
def add_labels(l_df):
    human_label = "'from': 'human', 'value':"
    gpt_label = "'from': 'gpt', 'value':"
    
    # Replace occurrences of the labels in the "conversations" column directly
    l_df["conversations"] = l_df["conversations"].replace(human_label, r'"role": "COUNSELEE", "text":', regex=True)
    l_df["conversations"] = l_df["conversations"].replace(gpt_label, r'"role": "THERAPIST", "text":', regex=True)
    
    return l_df

# Remove whitespace
def remove_whitespace(df):
    df["conversations"] = df["conversations"].replace(" ", "")
    
    return df

# Fix missing commas between dictionary entries in a list
def fix_missing_commas(dataframe):
    # Define the regex pattern to detect a missing comma between dictionaries
    pattern = r"(\}\s*\{)"
    
    # Apply the fix by adding a comma between the dictionaries where needed
    dataframe["conversations"] = dataframe["conversations"].apply(lambda x: re.sub(pattern, "}, {", str(x)))
    
    return dataframe

if __name__ == "__main__":
    df = input("Insert Dataset Name: ")
    main(df)

