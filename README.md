# Differentially Private Set Union (DPSU)

The code in this repository is part of the supplementary material for the following paper:

- **Title**: Incorporating Item Frequency for Differentially Private Set Union
- **Authors**: Ricardo Silva Carvalho, Ke Wang, Lovedeep Gondara
- **Conference**: AAAI 2022

**Problem:**

- In the experiments showed in this repository we consider the ubiquitous problem of building a vocabulary, which is equivalent to releasing the set union of n-grams for n=1 (unigrams). 

**Goal:**

- Our goal is to output the largest vocabulary possible while satisfying user-level Differential Privacy.


# Instructions

Please note that:

- The instructions on how to download the datasets used are detailed in `1_data_processing.ipynb`.
- You can use `1_data_processing.ipynb` to preprocess the datasets in order to run DPSU.
- You can go to `2_run_dpsu.ipynb` to run the DPSU mechanisms for the datasets used.
