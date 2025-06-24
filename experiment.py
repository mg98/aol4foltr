from aol4foltr.foltr.foltr_async import FOLTRAsync
from aol4foltr.foltr.foltr_sync import FOLTRSync
from aol4foltr.data.metadata import Metadata
from aol4foltr.data.letor import AOL4FOLTRDataset
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def train_test_split(df: pd.DataFrame, train_ratio: float = 0.8):
    # Get unique query IDs and split into train/test
    unique_qids = df['qid'].unique()
    train_size = int(len(unique_qids) * 0.8)
    train_qids = unique_qids[:train_size]
    test_qids = unique_qids[train_size:]

    # Split metadata into train and test based on query IDs
    train_df = df[df['qid'].isin(train_qids)]
    test_df = df[df['qid'].isin(test_qids)]

    return train_df, test_df

def redistribute_iid(df: pd.DataFrame):
    df = df.copy()
    user_ids = df['user_id'].unique()
    q_per_user = len(df) // len(user_ids)
    remainder = len(df) % len(user_ids)

    # Create the user_id assignment list
    user_id_assignments = np.repeat(user_ids, q_per_user)
    if remainder > 0:
        user_id_assignments = np.concatenate([user_id_assignments, user_ids[:remainder]])

    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    df['user_id'] = user_id_assignments

    return df

batch_size = 4
lr = 0.1
sensitivity = 5
epsilon = 4.5
enable_noise = True
max_rounds = 100

print("Loading metadata...")
metadata_df = pd.read_csv('dataset/metadata.csv', parse_dates=['timestamp'], low_memory=False)
metadata_df = metadata_df.sort_values('timestamp')

# Get top 100 users by number of queries
user_query_counts = metadata_df.groupby('user_id')['qid'].nunique()
top_users = user_query_counts.nlargest(100).index
metadata_df = metadata_df[metadata_df['user_id'].isin(top_users)]
metadata_df = metadata_df[metadata_df['user_id'] != 71845]

train_df, test_df = train_test_split(metadata_df)
train_data = Metadata(train_df)
test_data = Metadata(test_df)

print("Loading letor dataset...")
letor_ds = AOL4FOLTRDataset('dataset/letor_100.txt')

print("Starting experiment...")

# Real data experiments
foltr_sync = FOLTRSync(train_data, test_data, letor_ds)
foltr_async = FOLTRAsync(train_data, test_data, letor_ds)

# IID data experiments
iid_metadata_df = redistribute_iid(metadata_df)

iid_train_df, iid_test_df = train_test_split(iid_metadata_df)
iid_train_data = Metadata(iid_train_df)
iid_test_data = Metadata(iid_test_df)

iid_foltr_sync = FOLTRSync(iid_train_data, iid_test_data, letor_ds)
iid_foltr_async = FOLTRAsync(iid_train_data, iid_test_data, letor_ds)

# Run all four experiments in parallel
results = Parallel(n_jobs=4)(delayed(experiment.run)(
    batch_size, max_rounds) for experiment in [
        foltr_async,
        foltr_sync,
        iid_foltr_async, 
        iid_foltr_sync
    ])

mrr_real_async, mrr_real_sync, mrr_iid_async, mrr_iid_sync = results

# Determine max length to pad shorter arrays
max_length = max(len(arr) for arr in results)

# Create results dictionary
results_dict = {
    'round': list(range(1, max_length + 1)),
    'mrr_real_async': mrr_real_async + [np.nan] * (max_length - len(mrr_real_async)),
    'mrr_real_sync': mrr_real_sync + [np.nan] * (max_length - len(mrr_real_sync)), 
    'mrr_iid_async': mrr_iid_async + [np.nan] * (max_length - len(mrr_iid_async)),
    'mrr_iid_sync': mrr_iid_sync + [np.nan] * (max_length - len(mrr_iid_sync))
}

# Save results to CSV
results_df = pd.DataFrame(results_dict)
results_df.to_csv('results/experiment_results.csv', index=False)
