import pandas as pd
import lmdb
import pickle
import numpy as np
from ltr.data import write_to_disk
from ltr.types import ClickThroughRecord
from fpdgd.client.client import RankingClient
from fpdgd.client.federated_optimize import average_mrr_at_k
from fpdgd.ranker.PDGDLinearRanker import PDGDLinearRanker
from fpdgd.data.LetorDataset import LetorDataset
from tqdm import tqdm
import os
from joblib import Parallel, delayed

top100_df = pd.read_csv('dataset/aol_dataset_top100.csv', parse_dates=['time']).sort_values('time')

class WeightedFeedback:
    def __init__(self, original_feedback, weighted_interactions):
        self.gradient = original_feedback.gradient
        self.parameters = original_feedback.parameters
        self.n_interactions = weighted_interactions

class Dataset:
    def __init__(self, batch_size=4, qids: list[str] = None, iid = False):
        self.df = top100_df.copy()
        self.df = self.df[self.df['query_id'].astype(str).isin(qids)]

        if iid:
            # For IID, ignore time and user_id - just randomly split all data into batches
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            self.df['batch'] = np.arange(len(self.df)) // batch_size
            
            # Only keep complete batches
            batch_sizes = self.df.groupby('batch').size()
            valid_batches = batch_sizes[batch_sizes == batch_size].index
            self.df = self.df[self.df['batch'].isin(valid_batches)]
            
            # Set dummy time and user_id for grouping
            self.df['batch_end'] = self.df['batch'] 
            self.df['user_id'] = 'iid_user'
            
            # Set events_df as grouped by dummy user and batch
            self.events_df = self.df.groupby(['user_id', 'batch'], sort=False)
            return
        
        dfs = []
        for user_id, group in self.df.groupby('user_id'):
            group = group.sort_values('time')
            group = group.copy()
            group['cum'] = range(len(group))
            group['batch'] = group['cum'] // batch_size

            # Only keep complete batches
            batch_sizes = group.groupby('batch').size()
            valid_batches = batch_sizes[batch_sizes == batch_size].index
            group = group[group['batch'].isin(valid_batches)]

            # Calculate batch end times
            group['batch_end'] = group.groupby('batch')['time'].transform('max')

            dfs.append(group)
        
        batch_df = pd.concat(dfs, ignore_index=True).sort_values('batch_end')
        self.events_df = batch_df.groupby(['user_id', 'batch'], sort=False)
    
    @property
    def user_ids(self):
        return set(name for name, _ in self.events_df.groups)


n_features = 103
batch_size = 4
lr = 0.1
sensitivity = 5
epsilon = 4.5
enable_noise = True

blacklisted_features = list(range(18, 33))+list(range(75, 78))+list(range(93, 96))
traindata = LetorDataset('dataset/train.txt', n_features, query_level_norm=True, blacklisted_features=blacklisted_features)
testdata = LetorDataset('dataset/test.txt', n_features, query_level_norm=True, blacklisted_features=blacklisted_features)
train_qids = [str(qid) for qid in traindata.get_all_querys()]
ds = Dataset(batch_size=batch_size, qids=train_qids, iid=False)
random_ds = Dataset(batch_size=batch_size, qids=train_qids, iid=True)

def run_experiment(ds: Dataset, sync=False, iid=False):
    """Run federated learning experiment"""
    ranker = PDGDLinearRanker(n_features, lr)
    mrr_server = []

    clients: dict[str, RankingClient] = {}
    for client_id, user_id in enumerate(ds.user_ids):
        client_ranker = PDGDLinearRanker(n_features, lr)
        client = RankingClient(traindata, client_ranker, client_id, sensitivity, epsilon, enable_noise, len(ds.user_ids))
        client.update_model(ranker)
        clients[user_id] = client

    client_last_update = {user_id: 0 for user_id in clients.keys()}

    max_events = min(10000, len(ds.events_df))

    if sync:
        for event_idx in tqdm(range(max_events), total=max_events, desc="Sync"):
            # before round: sync client models
            for client in clients.values():
                client.update_model(ranker)

            # collect feedbacks
            feedbacks = []
            for user_id, group in ds.df.groupby('user_id'):
                group = group.copy()
                group.sort_values('time')
                qids = group.iloc[event_idx*batch_size:(event_idx+1)*batch_size]['query_id'].unique().tolist()
                qids = [str(qid) for qid in qids]
                if len(qids) < batch_size:
                    continue
                client = clients[user_id]
                client_message, client_metric = client.client_ranker_update_queries(qids)
                feedbacks.append(WeightedFeedback(client_message, client_message.n_interactions))
            
            if len(feedbacks) == 0:
                break

            # update global model
            ranker.federated_averaging_weights(feedbacks)

            # evaluate global model
            all_result = ranker.get_all_query_result_list(testdata)
            mrr = average_mrr_at_k(testdata, all_result, 20)
            mrr_server.append(mrr)

    else:
        # ASYNC

        for event_idx, ((user_id, batch_num), event_group) in tqdm(enumerate(ds.events_df), total=max_events, desc="Async"):
            if event_idx >= max_events:
                break

            assert len(event_group['user_id'].unique()) == 1, "Event group contains multiple user IDs"
            assert event_group['user_id'].iloc[0] == user_id, "Event group user ID does not match expected user ID"
            assert len(event_group['query_id'].unique()) == batch_size, "batch size mismatch"

            queries_in_event = [str(qid) for qid in event_group['query_id'].unique().tolist()]
            assert len(queries_in_event) == batch_size, "batch size mismatch"
            client = clients[user_id]

            client_message, client_metric = client.client_ranker_update_queries(queries_in_event)
            client_staleness = event_idx + 1 - client_last_update[user_id]
            client_last_update[user_id] = event_idx + 1
            
            ranker.async_federated_averaging_weights(
                (client_message.gradient, client_message.parameters, client_message.n_interactions, client_staleness)
                )
            client.update_model(ranker)
            
            all_result = ranker.get_all_query_result_list(testdata)
            mrr = average_mrr_at_k(testdata, all_result, 20)
            mrr_server.append(mrr)

    return mrr_server

# Run all four experiments

results = Parallel(n_jobs=4)(delayed(run_experiment)(
    dataset, sync=sync, iid=iid) for dataset, sync, iid in [
        (ds, False, False),
        (ds, True, False), 
        (random_ds, False, True),
        (random_ds, True, True)
    ])

mrr_real_async, mrr_real_sync, mrr_iid_async, mrr_iid_sync = results

print("Done")

# Export results to CSV

# Determine the maximum length to pad shorter arrays
max_length = max(len(mrr_real_async), len(mrr_real_sync), len(mrr_iid_async), len(mrr_iid_sync))

# Create a dictionary to store all results
results_dict = {
    'event_index': list(range(1, max_length + 1))
}

# Pad arrays with NaN if they're shorter than max_length
def pad_array(arr, target_length):
    if len(arr) < target_length:
        return arr + [np.nan] * (target_length - len(arr))
    return arr[:target_length]

results_dict['mrr_real_async'] = pad_array(mrr_real_async, max_length)
results_dict['mrr_real_sync'] = pad_array(mrr_real_sync, max_length)
results_dict['mrr_iid_async'] = pad_array(mrr_iid_async, max_length)
results_dict['mrr_iid_sync'] = pad_array(mrr_iid_sync, max_length)

# Create DataFrame and export to CSV
results_df = pd.DataFrame(results_dict)
results_df.to_csv('results/mrr_results.csv', index=False)
print("Results exported to mrr_results.csv")
