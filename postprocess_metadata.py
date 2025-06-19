import os
import pandas as pd

df = pd.read_csv('dataset/metadata_raw.csv', parse_dates=['timestamp'], low_memory=False)
df = df.sort_values('timestamp')
df.index.name = 'qid'
df.to_csv('dataset/metadata.csv', index=True)
os.remove('dataset/metadata_raw.csv')

print("Success!")
