import pandas as pd
from ast import literal_eval

class Metadata(pd.DataFrame):
    def __init__(self, src: str | pd.DataFrame):
        if isinstance(src, str):
            df = pd.read_csv('dataset/metadata.csv', parse_dates=['timestamp'], low_memory=False)
            super().__init__(df)
        elif isinstance(src, pd.DataFrame):
            src = src.copy()
            src['timestamp'] = pd.to_datetime(src['timestamp'])
            super().__init__(src)
        else:
            raise ValueError(f'Invalid metadata source: {type(src)}')

    @property
    def user_ids(self):
        return self['user_id'].unique()
