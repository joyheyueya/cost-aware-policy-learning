import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os

df = pd.read_csv('ExperimentData/Social/ProcessedData/socialpressnofact.csv')
df['age'] = [2006]*len(df) - df['yob']
df['treat_nothing'] = 1 - df['treatment_dum']
df['treat_letter'] = df['treatment_dum']

interaction_terms = [
    ['age', 'treat_civic'],
    ['age', 'treat_neighbors'],
    ['p2000', 'treat_nothing'],
    ['p2000', 'treat_neighbors'],
    ['sex', 'treat_civic'],
    ['sex', 'treat_nothing'],
    ['hh_size', 'treat_nothing'],
    ['hh_size', 'treat_self'],
]

for t in interaction_terms:
    df[t[0] + '_' + t[1]] = df[t[0]]*df[t[1]]

def get_self_feature(x):
    if x['treat_self'] == 0:
        if x['treat_neighbors'] == 1:
            return 1
        else: 
            return 0
    else:
        return x['treat_self']
df['treat_self_feature'] = df.apply(get_self_feature, axis=1)

unique_cities = df['city'].unique()
train_cities, test_cities = train_test_split(unique_cities, test_size=0.1, random_state=42)

train_df = df[df['city'].isin(train_cities)]
test_df = df[df['city'].isin(test_cities)]
print(len(train_df))
print(len(test_df))

context_feature_list = ['sex','age','g2000', 'g2002', 'g2004', 'p2000', 'p2002', 'p2004', 'hh_size']
interaction_feature_list = ['age_treat_civic', 'age_treat_neighbors', 'p2000_treat_nothing',
       'p2000_treat_neighbors', 'sex_treat_civic', 'sex_treat_nothing',
       'hh_size_treat_nothing', 'hh_size_treat_self']
action_feature_list = ['treat_nothing', 'treat_letter', 'treat_hawthorne', 'treat_civic', 'treat_neighbors', 'treat_self_feature']

action_matrix = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 1],
    [0, 1, 0, 0, 1, 0]
])
action_feature_matrix = np.eye(5)

action_names = ['treat_nothing', 'treat_hawthorne', 'treat_civic', 'treat_neighbors', 'treat_self']

costs = np.array([0.1, 0.5, 0.5, 10, 2.5])

def process_dataframe(df):
    processed = {}
    for i, grouped_df in df.groupby(np.arange(len(df)) // 5):
        contexts = grouped_df.iloc[:, :-2].values
        ranks = grouped_df['outcome_voted'].values
        costs = grouped_df['cost'].values
        processed[i] = (contexts, ranks, costs)
    return processed


def get_expanded_df(df):
    expanded_data = []
    for index, row in df.iterrows():
        base_context = row[context_feature_list].values
        outcome_voted = row['outcome_voted']

        for action_idx, action_row in enumerate(action_matrix):
            # Set ranks for only the correct action
            ranks = [np.nan] * 5
            if row[action_names[action_idx]] == 1:
                correct_index = action_idx
                ranks[correct_index] = outcome_voted

            interaction_features = np.array([
                row[term[0]] * action_feature_matrix[action_idx][action_names.index(term[1])]
                for term in interaction_terms
            ])
#             if action_idx ==1:
#                 return interaction_features, action_row

            full_context = np.concatenate([base_context, interaction_features, action_row])
            expanded_data.append(np.append(full_context, [ranks[action_idx], costs[action_idx]]))

    interaction_feature_names = [t[0] + '_' + t[1] for t in interaction_terms]
    column_names = context_feature_list + interaction_feature_names + action_feature_list + ['outcome_voted', 'cost']
    expanded_df = pd.DataFrame(expanded_data, columns=column_names)
    return expanded_df

expanded_df = get_expanded_df(train_df)
TRAIN_PROCESSED = process_dataframe(expanded_df)

def create_folder(RESULTS_FOLDER):
    # Check if the folder exists
    if not os.path.exists(RESULTS_FOLDER):
        # If it does not exist, create it
        os.makedirs(RESULTS_FOLDER)
        print("Folder created:", RESULTS_FOLDER)
    else:
        print("Folder already exists:", RESULTS_FOLDER)  

DATA_DIR = 'voting_data/'
create_folder(DATA_DIR)
with open(DATA_DIR + 'train_processed_n' + str(costs[0]).replace('.','') + '.pkl', 'wb') as f:
    pickle.dump(TRAIN_PROCESSED, f)

expanded_df = get_expanded_df(test_df)
TRAIN_PROCESSED = process_dataframe(expanded_df)
with open(DATA_DIR + 'test_processed_n' + str(costs[0]).replace('.','') + '.pkl', 'wb') as f:
    pickle.dump(TRAIN_PROCESSED, f)