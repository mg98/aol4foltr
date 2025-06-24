# AOL4FOLTR

This repository contains the source code to generate the dataset and results presented in the paper _"AOL4FOLTR: A Large-Scale Web Search Dataset for Federated Online Learning to Rank"_ (currently under review).

**AOL4FOLTR** is a dataset specifically tailored with its use in _Federated Online Learning-to-Rank_ (short: _FOLTR_) in mind.
It contains raw search queries and document contents, user IDs, and timestamps, based on [AOL-IA](https://ir-datasets.com/aol-ia.html), and originally, the 2006 AOL query logs.
Furthermore, we generated top-20 result lists for each query, and designed 103 features to enable learning-to-rank.

## Links

- [Download Dataset](https://zenodo.org/records/15689455)
- [Generate Dataset](./docs/generate_dataset.md)
- [How to Use Dataset](./docs/howto.md)
- [Reproduce Results](./docs/reproduce.md)
- [Learning-to-Rank Feature List](./docs/feature_list.md)

## Acknowledgments

Our implementation of FPDGD is based on the code in <https://github.com/ielab/fpdgd-ictir2021>.
