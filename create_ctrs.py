import random
import torch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from ltr.data import compile_clickthrough_records
import pickle
import lmdb
import warnings
import argparse
import ast

warnings.filterwarnings(
    "ignore",
    message="A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.",
    category=UserWarning,
    module="joblib.externals.loky.process_executor"
)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
if hasattr(torch, 'mps'):
    torch.mps.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process AOL dataset with start and end indices')
    parser.add_argument('--job-id', type=int, help='Slurm job ID')
    parser.add_argument('--job-count', type=int, help='Total number of jobs')
    args = parser.parse_args()
    
    df = pd.read_csv('dataset/metadata.csv', parse_dates=['timestamp'], low_memory=False)
    
    if args.job_id is not None and args.job_count is not None:
        df = df[df.index % args.job_count == args.job_id].copy()
        print(f"Processing {len(df)} rows... (job {args.job_id+1}/{args.job_count})")
    else:
        print(f"Processing {len(df)} rows...")

    df['candidate_doc_ids'] = df['candidate_doc_ids'].apply(ast.literal_eval)
    
    query_ctrs = compile_clickthrough_records(df, True)

    print("Writing results to LMDB...")

    db_path = f'dataset/ctrs_{args.job_id+1}.lmdb' if args.job_id is not None else 'dataset/ctrs.lmdb'
    with lmdb.open(db_path, map_size=2**40) as db:
        with db.begin(write=True) as txn:
            for query_id, ctrs in tqdm(query_ctrs.items(), total=len(query_ctrs), desc="Writing to LMDB"):
                first_qid = int(ctrs[0][1])
                assert all(ctr[1] == first_qid for ctr in ctrs), f"Query ID mismatch within CTRs for query {query_id}"
                assert all(ctr[1] == int(query_id) for ctr in ctrs), f"Query ID mismatch for query {first_qid} != {query_id}"
                txn.put(str(query_id).encode(), pickle.dumps(ctrs))
    
    print(f"Saved to disk at {db_path}")
