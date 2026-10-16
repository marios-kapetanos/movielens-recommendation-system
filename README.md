# MovieLens 10M – Movie Recommendation System

An end-to-end Machine Learning project exploring movie ratings, collaborative filtering, and personalized recommendation strategies using the MovieLens 10M dataset.

Developed as part of the **AI Data Factory – Machine Learning & Data Analysis Bootcamp at Athens University of Economics and Business (AUEB)**.

## Project Overview

This project analyzes more than 10 million movie ratings and develops a recommendation system based on Matrix Factorization with Stochastic Gradient Descent (SGD).

It also investigates the cold-start problem and implements Bayesian and content-based recommendation strategies.

## Technologies

- Python
- Pandas & NumPy
- Matplotlib
- Scikit-learn
- Numba
- Jupyter Notebook

## Methodology

**1. Data Preparation**

Loaded and merged movie metadata and user ratings, extracting the year of each rating.

**2. Exploratory Data Analysis**

Investigated changes in average movie ratings across genres and examined the influence of sample size on rating estimates.

**3. Matrix Factorization**

Implemented collaborative filtering using SGD, latent factors, user and movie biases, and L2 regularization.

**4. Model Evaluation**

Used a temporal split to train on ratings before 2008 and evaluate on ratings from 2008–2009.

| Metric | Result |
|---|---:|
| Test MSE | 0.8209 |
| Test RMSE | 0.9060 |
| Global Mean Baseline RMSE | 0.9990 |
| RMSE Improvement | 9.30% |

**5. Cold-Start Recommendations**

Generated movie recommendations for new users using Bayesian weighted ratings.

**6. Personalized Recommendations**

Developed a content-based recommendation strategy using movie genre similarity and rating quality, based on preferences for *Iron Man*, *300*, and *Transformers*.

## Dataset

Dataset: MovieLens 10M, provided by GroupLens Research.

https://grouplens.org/datasets/movielens/

Download and extract the dataset, placing `movies.dat` and `ratings.dat` in the same directory as the Jupyter Notebook.

The dataset is not included in this repository.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Then open `movielens_recommender_final_corrected.ipynb` in Jupyter Notebook or VS Code and run the cells sequentially.

## Limitations and Future Improvements

- Cold-start users limit collaborative filtering performance.
- Recommendation weights are heuristic rather than learned.
- Hyperparameter tuning could improve predictive accuracy.
- Ranking metrics and diversity-aware recommendations could provide a more complete evaluation.

## Key Learning Outcomes

- Processing large-scale rating datasets
- Building collaborative filtering models
- Applying stochastic gradient descent
- Evaluating predictive models with MSE and RMSE
- Addressing cold-start recommendation scenarios
- Combining collaborative and content-based approaches
