import pandas as pd
import lmdb
from ltr.types import ClickThroughRecord
from ltr.data import write_to_disk
import pickle
from tqdm import tqdm
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, help="path to dataset csv")
parser.add_argument('--k', type=int, help="filter top-k users")
parser.add_argument('--output', type=str, help="path to output file")
args = parser.parse_args()

random.seed(42)

db = lmdb.open('dataset/ctrs.lmdb', readonly=True)

df = pd.read_csv(args.ds, parse_dates=['timestamp'])
if args.k is not None:
    user_query_counts = df.groupby('user_id')['qid'].nunique()
    top_users = user_query_counts.nlargest(args.k).index
    df = df[df['user_id'].isin(top_users)]
df = df.sort_values('timestamp')
# Calculate cutoff index for 80% of data
# cutoff_idx = int(len(df) * 0.8)
# df = df.iloc[cutoff_idx:]

with db.begin() as txn:
    with open(args.output, 'w') as f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            key = str(row['qid']).encode()
            value = txn.get(key)
            if value is None:
                print(f"No record found for qid {row['qid']}")
                continue
            
            query_ctr_arrays = pickle.loads(value)
            query_ctrs = [ClickThroughRecord.from_array(ctr_array) for ctr_array in query_ctr_arrays]
            assert all(ctr.qid == row['qid'] for ctr in query_ctrs), f"Query ID mismatch for query {row['qid']}"

            for ctr in query_ctrs:
                f.write(str(ctr) + '\n')

print("Done")
