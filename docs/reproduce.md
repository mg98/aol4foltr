# Reproduce Experimental Results

This repository contains scripts for reproducing the results stated in the paper.
This includes the dataset analysis (Section 4) and the FOLTR simulation with 100 clients (Section 5).

The results encompass

- Basic statistics
- Data quantity (queries per user)
- Temporal patterns
- Feature distribution divergence
- FOLTR simulation

## Requirements

Make sure to have the dataset either downloaded or [generated from sources](./generate_dataset.md) in the `dataset/` directory.
All analyses and experiments can be run on consumer-grade hardware.
The most expensive workloads still finished within 1 hour on a MacBook M2 Max.
We use R for analytics and plotting.

## Generate Results

To generate results for feature distribution divergence, run:

```
python measure_feat_div.py
```

To generate results for FOLTR simulation, run:

```
python experiment.py
```

The remaining results can be extracted from the dataset itself (i.e., `metadata.csv`).

## Show Results

Analytics and plotting is done with R.

```
Rscript analysis.R
```

Basic statistics are printed to console. The rest is exported as both TEX and PDF to `results/`.
