from fpdgd.data.LetorDataset import LetorDataset
import pandas as pd

class AOL4FOLTRDataset(LetorDataset):
    
    N_FEATURES = 103

    def __init__(self, path):
        global_features = list(range(18, 33))+list(range(75, 78))+list(range(93, 96))
        super().__init__(path, self.N_FEATURES, query_level_norm=True, global_level_norm=global_features)
