import os
import pandas as pd
import numpy as np
from ltr.types import ClickThroughRecord, FeatureVector
from joblib import Parallel, delayed
from tqdm import tqdm
import ir_datasets
from ir_datasets.datasets.aol_ia import AolIaDoc

def compile_clickthrough_records(
        df: pd.DataFrame, 
        parallel: bool = False) -> dict[str, list[np.ndarray]]:
    """
    Compile raw clickthrough records into LTR-format records encoded as arrays.
    Output maps query_id to a list of array-encoded ClickThroughRecords.

    Args:
        df: Dataframe containing the clickthrough records
        parallel: Whether to use parallel processing for acceleration
    """

    def process_query(row: pd.Series) -> list[np.ndarray]:
        """
        Transform each query (row) into a list of ClickThroughRecord arrays (one per candidate document).
        """
        dataset = ir_datasets.load("aol-ia")
        docs_store = dataset.docs_store()
        candidate_docs: list[AolIaDoc] = []

        # Create candidate docs
        for docid in row['candidate_doc_ids']:
            doc = docs_store.get(docid)
            candidate_docs.append(doc)

        # Create clickthrough records for this row
        ctr_arrays: list[np.ndarray] = []
        for doc in candidate_docs:
            feats = FeatureVector.make(candidate_docs, doc, row['query'])
            ctr = ClickThroughRecord(doc.doc_id == row['doc_id'], row['query_id'], feats)
            ctr_arrays.append(ctr.to_array())
        
        return ctr_arrays
    
    if parallel:
        query_ctrs = Parallel(n_jobs=-1, batch_size=1024)(
            delayed(process_query)(query_row) for _, query_row in tqdm(df.iterrows(), total=len(df), desc="Processing rows for CTRs")
        )
        ctrs = {str(query_row['query_id']): ctr_arrays for (_, query_row), ctr_arrays in zip(df.iterrows(), query_ctrs)}
    else:
        ctrs = {str(query_row['query_id']): process_query(query_row) 
                for _, query_row in tqdm(df.iterrows(), total=len(df), desc="Processing rows for CTRs")}
    return ctrs

def write_to_disk(ctrs: list[ClickThroughRecord], path: str):
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'wb') as f:
        for ctr in ctrs:
            f.write(str(ctr).encode() + b'\n')

