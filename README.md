# Differentially Private Set Union

The code in this submission is part of the supplementary material for the following paper:

- **Title**: Incorporating Item Frequency for Differentially Private Set Union


Problem:

- We consider the ubiquitous problem of building a vocabulary, which is equivalent to releasing the set union of n-grams for n=1 (unigrams). 

Goal:

- Our goal is to output the largest vocabulary possible while satisfying user-level Differential Privacy.



# Instructions

The currently submitted zip file contains:

- Preprocessed *sensitive* dataset 'finance' at `data/finance_cleaned.csv`
- Preprocessed *public* dataset 'imdb' at `data/imdb_cleaned.csv`
- Notebook to preprocess any dataset: `1_data_processing.ipynb`
- Notebook to run DPSU: `2_run_dpsu.ipynb`

Please note that:

- You do **not** have to download any dataset to run the DPSU algorithms.
- You can go directly to `2_run_dpsu.ipynb` and run the mechanisms for the datasets provided here, i.e. datasets 'finance' and 'imdb'.
- Only use `1_data_processing.ipynb` if you want to preprocess any additional dataset aside from 'finance' and 'imdb' to run DPSU.
- The instructions on how to download the other datasets to preprocess are detailed in `1_data_processing.ipynb`.
- This submission only contains one sensitive dataset and one public dataset due to the maximum file size in CMT being 100MB.
