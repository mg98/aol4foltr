import os
import pandas as pd

df = pd.read_csv('aol_dataset_raw.csv')
df = df.sort_values('time')
# df = df.drop_duplicates(subset=['user_id', 'query'], keep='last')

df.index.name = 'query_id'
df.to_csv('aol_dataset.csv', index=True)
os.remove('aol_dataset_raw.csv')

print("Success!")
