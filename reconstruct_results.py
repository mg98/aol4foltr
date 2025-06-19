import pandas as pd
import argparse
from tqdm import tqdm
import ir_datasets
from pyserini.search.lucene import LuceneSearcher
import random
from collections import defaultdict

parser = argparse.ArgumentParser(description='Process AOL dataset with start and end indices')
parser.add_argument('--job-id', type=int, help='Slurm job ID')
parser.add_argument('--job-count', type=int, help='Total number of jobs')
parser.add_argument('--k', type=int, default=20, help='Window size for candidate documents')
args = parser.parse_args()

dataset = ir_datasets.load("aol-ia")
searcher = LuceneSearcher('indexes/docs_jsonl')

docs_store = dataset.docs_store()

qlogs = []
for qlog in tqdm(dataset.qlogs_iter(), total=dataset.qlogs_count(), desc="Prepare qlogs"):
    if qlog.query.strip() == '':
        continue
    if len(qlog.items) != 1:
        continue
    qlogs.append({
        'query_id': qlog.query_id,
        'query': qlog.query.strip().lower(),
        'timestamp': qlog.time,
        'user_id': qlog.user_id,
        'target_doc_id': qlog.items[0].doc_id,
    })

qlogs = pd.DataFrame(qlogs)
user_query_counts = qlogs.groupby('user_id').size()
top_users = user_query_counts.nlargest(10000).index
qlogs = qlogs[qlogs['user_id'].isin(top_users)]

# Build index mapping query to set of target doc_ids
natural_candidates = qlogs.groupby('query_id')['target_doc_id'].agg(set).to_dict()

def process_qlog(searcher, qlog):
    # 1. Sample up to k known target docs
    docs = set(random.sample(
        list(natural_candidates[qlog.query_id] - {qlog.target_doc_id}), 
        min(args.k, len(natural_candidates[qlog.query_id] - {qlog.target_doc_id})))
    )

    # 2. Ensure real target is included
    if qlog.target_doc_id not in docs:
        if docs: docs.pop()
        docs.add(qlog.target_doc_id)

    alt_candidates = docs - {qlog.target_doc_id}

    # 3. Find real target within top-k
    if len(docs) < 20:
        # top 1000 excluding alt_candidates
        search_results = list(set(hit.docid for hit in searcher.search(qlog.query, k=1000) if hit.docid not in alt_candidates))
        if qlog.target_doc_id not in search_results or len(search_results) < args.k - len(docs):
            return None
        
        # Apply window of size k with target document at random position
        pos = search_results.index(qlog.target_doc_id)
        window_size = args.k - len(alt_candidates) # window of missing candidates
        if len(search_results) < window_size:
            # not enough candidates
            return None
        random_offset = random.randint(0, window_size - 1)
        start_pos = pos - random_offset
        end_pos = start_pos + window_size
        
        # Handle bounds - ensure we always have window_size documents
        if start_pos < 0:
            start_pos = 0
            end_pos = window_size
        elif end_pos > len(search_results):
            end_pos = len(search_results)
            start_pos = max(0, end_pos - window_size)
            
        docs.update(search_results[start_pos:end_pos])

    assert len(docs) == 20, f"Expected 20 docs, got {len(docs)}"
    assert qlog.target_doc_id in docs
    
    return {
        'user_id': qlog.user_id,
        'timestamp': qlog.timestamp,
        'query': qlog.query,
        'doc_id': qlog.target_doc_id,
        'candidate_doc_ids': docs
    }

results = []
for idx, qlog in tqdm(enumerate(qlogs.itertuples()), total=len(qlogs)):
    if args.job_id is not None and idx % args.job_count != args.job_id:
        continue
    result = process_qlog(searcher, qlog)
    if result is not None:
        results.append(result)

df = pd.DataFrame(results)

print(f"Saving DataFrame with {len(df)} records to disk...")

csv_filename = 'metadata_raw.csv'
if args.job_id is not None:
    csv_filename = f'metadata_{args.job_id}_{args.job_count}.csv'
df.to_csv('dataset/' + csv_filename, index=False)

print("Success!")
