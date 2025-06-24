# How To Use

This document explains how to load and use this dataset for FOLTR.

AOL4FOLTR consists of two files: `metadata.csv` and `letor.txt` (i.e., after decompression): both files are linked via the 'qid' attribute.
We intentionally used open standard formats to ensure broad accessibility and ease of use with popular libraries such as `pandas`.

It is important to note that our LETOR dataset contains raw feature values.
LTR models tend to learn more effectively when features are normalized, either on the level of the query or globally.

We provide two light-weight abstractions to facilitate FOLTR simulations and take care of the feature-wise normalization.

```python
from aol4foltr.data.metadata import Metadata
from aol4foltr.data.letor import AOL4FOLTRDataset

metadata = Metadata('dataset/metadata.csv')
letor_ds = AOL4FOLTRDataset('dataset/letor.txt')
```

For a full example of how to use this dataset, please refer to [experiment.py](../experiment.py).
