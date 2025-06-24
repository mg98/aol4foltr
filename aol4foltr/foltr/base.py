from abc import ABC, abstractmethod
from aol4foltr.data.letor import AOL4FOLTRDataset
from fpdgd.ranker.PDGDLinearRanker import PDGDLinearRanker
from fpdgd.client.client import RankingClient
from aol4foltr.data.metadata import Metadata

class FOLTRBase(ABC):
    """
    Base class for FOLTR.
    """

    def __init__(self, 
                 train_data: Metadata, 
                 test_data: Metadata,
                 letor_ds: AOL4FOLTRDataset, 
                 lr: float = 0.1,
                 sensitivity: float = 5.0,
                 epsilon: float = 4.5,
                 personalization_lambda: float = 1.0):
        
        enable_noise = sensitivity != 0 or epsilon != 0
        self.train_data = train_data
        self.test_data = test_data
        self.letor_ds = letor_ds
        self.ranker = PDGDLinearRanker(letor_ds.N_FEATURES, lr)
        self.clients: dict[str, RankingClient] = {}

        # Create clients for each user
        for client_id, user_id in enumerate(train_data.user_ids):
            client_ranker = PDGDLinearRanker(letor_ds.N_FEATURES, lr)
            client = RankingClient(
                dataset=letor_ds, 
                init_model=client_ranker, 
                seed=client_id, 
                sensitivity=sensitivity, 
                epsilon=epsilon, 
                enable_noise=enable_noise, 
                n_clients=len(train_data.user_ids),
                personalization_lambda=personalization_lambda
                )
            client.update_model(self.ranker)
            self.clients[user_id] = client
