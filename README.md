The dataset is large, so I put the link to the dataset here: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html

The results folder is also large, so I put the link to the google drive here: https://drive.google.com/drive/folders/1Xm6ACHM-KhLFY-7F2QLCv1wyyTnTcMeI?usp=drive_link

# DATA1030 Final Project – Amazon Video Game Review Rating Prediction

This repo contains the code and analysis for my DATA1030 final project.  
The goal is to predict **1–5 star ratings** for Amazon *Video Games* reviews using review text and a small set of numeric features (sentiment, review length, helpfulness, punctuation).

The main pieces are:

- `project.ipynb` – main notebook with:
  - data loading and exploratory data analysis (EDA)
  - group-aware train/val/test splits by `reviewerID`
  - shared preprocessing pipeline (TF–IDF + numeric features)
  - model training and tuning (Logistic Regression, LinearSVC, Random Forest, XGBoost)
  - evaluation (accuracy, macro/weighted F1, MAE, within-1 accuracy)
  - uncertainty analysis (across CV folds and random seeds)
  - global feature importance (coefficients, permutation, tree-based)
  - local explanations with SHAP
- `results/` – generated artifacts (tables, plots, saved models; large `.pkl` files are git-ignored).
- LaTeX report (in a separate Overleaf project) that uses the figures and tables from `results/`.

---

## Data

The project uses the **Amazon Video Games 5-core** review data released by Ni et al. (2019).  
The raw data is **not** included here due to size and licensing.

To run the notebook, download the appropriate file from the original source and place it under a local `data/` directory (e.g. `data/reviews_Video_Games_5.json`), then update the path in the notebook if needed.

---

## Environment and dependencies

The code was developed with:

- **Python**: 3.12.x  
- Main libraries:
  - `numpy`, `pandas`
  - `scikit-learn`
  - `xgboost`
  - `shap`
  - `matplotlib` (and optionally `seaborn`)
  - `textblob` (or similar sentiment library, if used)
  - `jupyterlab` / `ipykernel`

A conda environment file is provided as `environment.yml`.

To create and activate the environment:

```bash
conda env create -f environment.yml
conda activate data1030-project

