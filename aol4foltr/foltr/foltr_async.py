from tqdm import tqdm
import pandas as pd
from aol4foltr.foltr.base import FOLTRBase
from aol4foltr.data.metadata import Metadata
from fpdgd.data.LetorDataset import LetorDataset
from fpdgd.client.federated_optimize import average_mrr_at_k

class FOLTRAsync(FOLTRBase):
    def __init__(self, 
                 train_data: Metadata,
                 test_data: Metadata,
                 letor_ds: LetorDataset, 
                 lr: float = 0.1, 
                 sensitivity: float = 5.0, 
                 epsilon: float = 4.5, 
                 personalization_lambda: float = 1.0
                 ):
        super().__init__(train_data, test_data, letor_ds, lr, sensitivity, epsilon, personalization_lambda)
        self.client_last_update = {user_id: 0 for user_id in self.clients.keys()}

    
    def batch(self, batch_size: int) -> pd.DataFrame:
        """
        Batch the dataset into groups of `batch_size` events.
        """
        dfs = []
        
        for user_id, group in self.train_data.groupby('user_id'):
            group = group.sort_values('timestamp')
            group = group.copy()
            group['cum'] = range(len(group))
            group['batch'] = group['cum'] // batch_size

            # Only keep complete batches
            batch_sizes = group.groupby('batch').size()
            valid_batches = batch_sizes[batch_sizes == batch_size].index
            group = group[group['batch'].isin(valid_batches)]

            # Calculate batch end times
            group['batch_end'] = group.groupby('batch')['timestamp'].transform('max')

            dfs.append(group)

        return pd.concat(dfs, ignore_index=True).sort_values('batch_end').groupby(['user_id', 'batch'], sort=False)

    def run(self, batche_size: int, max_rounds: int|None = None) -> list[float]:
        batches = self.batch(batche_size)
        max_rounds = len(batches) if max_rounds is None else min(max_rounds, len(batches))
        test_qids = [str(qid) for qid in self.test_data['qid'].unique()]
        mrrs = []
        
        for round_idx, ((user_id, batch_num), event_group) in tqdm(enumerate(batches), total=max_rounds, desc="Async"):
            if round_idx >= max_rounds:
                break

            assert len(event_group['user_id'].unique()) == 1, "Event group contains multiple user IDs"
            assert event_group['user_id'].iloc[0] == user_id, "Event group user ID does not match expected user ID"

            queries_in_event = [str(qid) for qid in event_group['qid'].unique().tolist()]
            client = self.clients[user_id]

            # update client model
            client_msg, client_metric = client.client_ranker_update_queries(queries_in_event)

            # update global model
            client_staleness = round_idx + 1 - self.client_last_update[user_id]
            self.client_last_update[user_id] = round_idx + 1
            self.ranker.async_federated_averaging_weights(
                (client_msg.gradient, client_msg.parameters, client_msg.n_interactions, client_staleness)
                )

            # sync client model
            client.update_model(self.ranker)

            # evaluate global model
            all_result = self.ranker.get_all_query_result_list(self.letor_ds, test_qids)
            mrr = average_mrr_at_k(self.letor_ds, all_result, 20)
            mrrs.append(mrr)

        return mrrs