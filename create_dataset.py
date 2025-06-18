import pandas as pd
import argparse
from tqdm import tqdm
import ir_datasets
from pyserini.search.lucene import LuceneSearcher
import random
from time import time
from collections import defaultdict
from utils.cache import Cache

dataset = ir_datasets.load("aol-ia")

parser = argparse.ArgumentParser(description='Process AOL dataset with start and end indices')
parser.add_argument('--job-id', type=int, help='Slurm job ID')
parser.add_argument('--job-count', type=int, help='Total number of jobs')
parser.add_argument('--k', type=int, default=20, help='Window size for candidate documents')
args = parser.parse_args()

searcher = LuceneSearcher('indexes/docs_jsonl')

docs_store = dataset.docs_store()
query_to_full_results = {}

def build_query_to_docs_index():
    # Build index mapping query_id to set of target doc_ids
    # First build index of valid query IDs (excluding user 71845)
    valid_query_ids = set()
    for qlog in tqdm(dataset.qlogs_iter(), total=dataset.qlogs_count(), desc="Building query filter"):
        if qlog.query.strip() != '':
            valid_query_ids.add(qlog.query_id)

    # Then build query_to_docs index using only valid queries
    query_to_docs_index: dict[str, set[str]] = defaultdict(set)
    for qrel in tqdm(dataset.qrels_iter(), total=dataset.qrels_count(), desc="Building query index"):
        if qrel.query_id in valid_query_ids:
            try:
                docs_store.get(qrel.doc_id)
            except KeyError as e:
                continue
            query_to_docs_index[qrel.query_id].add(qrel.doc_id)

    return query_to_docs_index

cache = Cache()
query_to_docs = None
if cache.get('query_to_docs_index') is not None:
    query_to_docs = cache.get('query_to_docs_index')
else:
    query_to_docs = build_query_to_docs_index()
    cache.set('query_to_docs_index', query_to_docs)

def process_qlog(searcher, qlog):
    if len(qlog.items) != 1: return None

    target_doc = qlog.items[0].doc_id

    # 1. Sample up to k known target docs
    docs = set(random.sample(
        list(query_to_docs[qlog.query_id] - {target_doc}), 
        min(args.k, len(query_to_docs[qlog.query_id] - {target_doc})))
    )

    # 2. Ensure real target is included
    if target_doc not in docs:
        if docs: docs.pop()
        docs.add(target_doc)

    alt_candidates = docs - {target_doc}

    # 3. Find real target within top-k
    if len(docs) < 20:
        # top 1000 excluding alt_candidates
        search_results = list(set(hit.docid for hit in searcher.search(qlog.query, k=1000) if hit.docid not in alt_candidates))
        if target_doc not in search_results or len(search_results) < args.k - len(docs):
            return None
        
        # Apply window of size k with target document at random position
        pos = search_results.index(target_doc)
        window_size = args.k - len(alt_candidates) # window of missing candidates
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

    if len(docs) != 20:
        return None # temporary fix
        assert len(docs) == 20, f"Expected 20 docs, got {len(docs)}"
    assert target_doc in docs
    
    return {
        'user_id': qlog.user_id,
        'time': qlog.time,
        'query': qlog.query,
        'doc_id': target_doc,
        'candidate_doc_ids': docs
    }

results = []
for idx, qlog in tqdm(enumerate(dataset.qlogs_iter()), total=dataset.qlogs_count()):
    if args.job_id is not None and idx % args.job_count != args.job_id:
        continue
    if qlog.query.strip() == '':
        continue
    result = process_qlog(searcher, qlog)
    if result is not None:
        results.append(result)

df = pd.DataFrame(results)

print(f"Saving DataFrame with {len(df)} records to disk...")
csv_filename = f'aol_{args.job_id}_{args.job_count}.csv' if args.job_id is not None else 'aol_raw_dataset.csv'
df.to_csv(csv_filename, index=False)
print("Success!")
