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
parser.add_argument('--output', type=str, help="path to output file")
args = parser.parse_args()

random.seed(42)

db = lmdb.open('ctrs.lmdb', readonly=True)

df = pd.read_csv(args.ds, parse_dates=['time'])
df = df.sort_values('time')
# Calculate cutoff index for 80% of data
cutoff_idx = int(len(df) * 0.8)
df = df.iloc[cutoff_idx:]


with db.begin() as txn:
    with open(args.output, 'a+') as f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            key = str(row['query_id']).encode()
            value = txn.get(key)
            if value is None:
                print(f"No record found for query_id {row['query_id']}")
                continue
            
            query_ctr_arrays = pickle.loads(value)
            query_ctrs = [ClickThroughRecord.from_array(ctr_array) for ctr_array in query_ctr_arrays]
            assert all(ctr.qid == row['query_id'] for ctr in query_ctrs), f"Query ID mismatch for query {row['query_id']}"

            for ctr in query_ctrs:
                f.write(str(ctr) + '\n')

print("Done")
