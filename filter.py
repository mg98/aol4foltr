import os
import pandas as pd
import ast
import ir_datasets
from tqdm import tqdm

df = pd.read_csv('dataset/aol_dataset.csv', parse_dates=['time'])
df['candidate_doc_ids'] = df['candidate_doc_ids'].apply(lambda x: list(ast.literal_eval(x)))

dataset = ir_datasets.load("aol-ia")
docs_store = dataset.docs_store()

rows_to_drop = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering dataset"):
    for doc_id in row['candidate_doc_ids']:
        try:
            doc = docs_store.get(doc_id)
        except:
            # Document not found, mark row for removal
            rows_to_drop.append(idx)
            break

# Drop rows with invalid doc_ids
df = df.drop(rows_to_drop)

# Export filtered dataset
df.to_csv('dataset/aol_dataset_filtered.csv', index=False)
