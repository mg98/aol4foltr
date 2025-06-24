from tqdm import tqdm
import pandas as pd
from aol4foltr.foltr.base import FOLTRBase
from aol4foltr.data.metadata import Metadata
from fpdgd.data.LetorDataset import LetorDataset
from fpdgd.client.federated_optimize import average_mrr_at_k

class WeightedFeedback:
    def __init__(self, original_feedback, weighted_interactions):
        self.gradient = original_feedback.gradient
        self.parameters = original_feedback.parameters
        self.n_interactions = weighted_interactions

class FOLTRSync(FOLTRBase):
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
    
    def run(self, batch_size: int, max_rounds: int|None = None) -> list[float]:
        max_batches = self.train_data.groupby('user_id')['qid'].nunique().max() // batch_size
        max_rounds = max_batches if max_rounds is None else min(max_rounds, max_batches)
        test_qids = [str(qid) for qid in self.test_data['qid'].unique()]
        mrrs = []

        for round_idx in tqdm(range(max_rounds), desc="Sync"):
            # before round: sync client models
            for client in self.clients.values():
                client.update_model(self.ranker)

            # collect feedbacks
            feedbacks = []
            for user_id, group in self.train_data.groupby('user_id'):
                group = group.copy()
                group.sort_values('timestamp')
                qids = group.iloc[round_idx*batch_size:(round_idx+1)*batch_size]['qid'].unique().tolist()
                qids = [str(qid) for qid in qids]
                if len(qids) < batch_size:
                    continue
                client = self.clients[user_id]
                client_message, client_metric = client.client_ranker_update_queries(qids)
                feedbacks.append(WeightedFeedback(client_message, client_message.n_interactions))
            
            if len(feedbacks) == 0:
                break
            
            # update global model
            self.ranker.federated_averaging_weights(feedbacks)

            # evaluate global model
            all_result = self.ranker.get_all_query_result_list(self.letor_ds, test_qids)
            mrr = average_mrr_at_k(self.letor_ds, all_result, 20)
            mrrs.append(mrr)

        return mrrs