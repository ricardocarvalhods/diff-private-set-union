import pandas as pd
from tqdm import tqdm
import os.path

def get_sensitive_preprocessed_data(sensitive_name):
    dataset = 'data/' + sensitive_name + '_cleaned.csv'
    
    if os.path.isfile(dataset):
        input_df = pd.read_csv(dataset, index_col=0).dropna()

        df_all_grams = []
        for index, group in tqdm(input_df.groupby("author"), position=0, leave=True):
            posts = group["clean_text"]
            posts = [p.split(" ") for p in posts]
            words = [tokens for p in posts for tokens in p]
            all_grams = words
            df_all_grams.append(all_grams)

        return df_all_grams
    else:
        print("File {} does NOT exist. Skipping...\n".format(dataset))
        return None

def get_public_histogram(public_dataset_name):
    dataset = 'data/' + str(public_dataset_name) + '_cleaned.csv'
    
    if os.path.isfile(dataset):
        public_df = pd.read_csv(dataset)

        users = []
        grams = []
        for index, group in tqdm(public_df.groupby("author"), position=0, leave=True):
            posts = group["clean_text"]
            posts = [str(p).split(" ") for p in posts]
            words = [tokens for p in posts for tokens in p]
            all_grams = list(set(words))

            for ga in all_grams:
                users.append(index)
                grams.append(ga)

        user_gram = pd.DataFrame({'author': users,'grams': grams})

        histogram = user_gram.groupby(['grams']).author.nunique()
        histogram = histogram.to_dict()
        
        return histogram
    else:
        print("File {} does NOT exist. Skipping...\n".format(dataset))
        return None    
