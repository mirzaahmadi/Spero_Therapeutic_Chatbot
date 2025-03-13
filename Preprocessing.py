import pandas as pd

# Read in the training dataset
def main(df):
    unprocessed_df = pd.read_csv(df)
    lowercased_df = lowercasing(unprocessed_df)
    labeled_df = add_labels(lowercased_df)
    print(labeled_df)
    
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

if __name__ == "__main__":
    df = input("Insert Dataset Name: ")
    main(df)