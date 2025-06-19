# Generating Dataset

This document provides instructions on how to generat the AOL4FOLTR dataset.
Please be aware that due to the size of this dataset, this process requires a lot of CPU compute time.
Access to a computing cluster is highly recommended.
For this tutorial, we assume access to a SLURM cluster.

## Step 1: Download AOL-IA Dataset

This setup assumes the AOL-IOA dataset is completely downloaded and located in `~/.ir_datasets/aol-ia`.
If this is not the case, please follow the instructions in the [aolia-tools](https://github.com/terrierteam/aolia-tools) repository. _(estimated time: 2 days)_

## Step 2: Initialize Environment

Install project dependencies in an environment with **Python 3.10**.

```
conda create -n pyserini python=3.10
conda activate pyserini
make install
```

Our scripts also require **Java (JDK) 21**.
Make sure to download the right [distribution](https://www.oracle.com/java/technologies/javase/jdk21-archive-downloads.html) for your OS.

```
cd ~
wget https://download.oracle.com/java/21/archive/jdk-21.0.6_linux-x64_bin.tar.gz
tar -xvzf jdk-21.0.6_linux-x64_bin.tar.gz
rm jdk-21.0.6_linux-x64_bin.tar.gz
```

### Step 3: Compile Dataset from Sources

The final dataset consists of two files:

- `metadata.csv` (~1 GB)
- `letor.txt` (~55 GB)

1. **Indexing:** Creates an index of the entire corpus of AOL-IA. Output goes to a new folder `indexes/` (~7 GB). Folder can be deleted after dataset creation. _(estimated time: 15 minutes)_
2. **Reconstruct results:** Reconstructs top-k results for each query log.
3. **Postprocessing:** Merges created metadata files and creation of _qids_.
4. **Creating CTRs:** Creates _Clickthrough Records (CTRs)_ for every query-document pair. A CTR contain qid, relevance label, and computed features. Results are stored into an LMDB, where each qid maps to a list of `k` CTRs (for `k` candidate documents). The CTRs are stored as numpy arrays to save space.
5. **Merge CTRs:** Unifies created LMDBs into one.
6. **Write LETOR:** Writes dataset to disk in LETOR format. Records are created based on metadata input; LMDB is used for lookup.

You can run all steps in sequence by submitting the following job chain:

```
make 
```
