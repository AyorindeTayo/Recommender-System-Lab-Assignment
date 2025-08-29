# Recommender-System-Lab-Assignment
# Lab: Building a Simple Recommender System (Based on Chapter 2)

## Objectives

By the end of this lab, you will be able to: - Understand the difference
between Content-Based and Collaborative Filtering. - Build a simple
recommender system using Python and scikit-learn. - Evaluate
recommendations using similarity measures.

------------------------------------------------------------------------

## Prerequisites

-   Python 3.x\

-   Install required libraries:

    ``` bash
    pip install pandas scikit-learn
    ```

------------------------------------------------------------------------

## Dataset

We will use a small **Movie Dataset** with the following structure:

  user_id   movie_title        rating
  --------- ------------------ --------
  1         Toy Story          5
  1         Jumanji            3
  2         Toy Story          4
  2         Grumpier Old Men   5
  3         Jumanji            4

👉 Create a CSV file named `movies.csv` with sample data like above.

------------------------------------------------------------------------

## Exercise Steps

### 1. Load the Dataset

``` python
import pandas as pd

# Load dataset
df = pd.read_csv("movies.csv")
print(df.head())
```

------------------------------------------------------------------------

### 2. Create a User-Item Matrix

``` python
user_item_matrix = df.pivot_table(index="user_id", columns="movie_title", values="rating")
print(user_item_matrix)
```

------------------------------------------------------------------------

### 3. Apply Collaborative Filtering (Cosine Similarity)

``` python
from sklearn.metrics.pairwise import cosine_similarity

# Fill NaN with 0 for similarity calculation
matrix_filled = user_item_matrix.fillna(0)

# Compute similarity between users
similarity = cosine_similarity(matrix_filled)
print("User Similarity Matrix:\n", similarity)
```

------------------------------------------------------------------------

### 4. Make Recommendations

``` python
# Example: Recommend for user 1 based on most similar user
import numpy as np

user_index = 0  # user_id = 1
similar_users = similarity[user_index]

# Find the most similar user (excluding self)
most_similar_user = np.argsort(similar_users)[-2]

# Get movies rated by most similar user
recommended_movies = user_item_matrix.iloc[most_similar_user].dropna().index.tolist()
print(f"Recommended movies for User 1: {recommended_movies}")
```

------------------------------------------------------------------------

### 5. Experiment


-   Try **Content-Based Filtering** using movie genres.\
-   Compare results from different approaches.

------------------------------------------------------------------------

## Deliverables

-   Submit your `movies.csv` dataset.\
-   Upload your Python notebook or `.py` script with code and results.\
-   Write a short reflection on:
    -   Which method (Content-Based vs Collaborative) worked better?
    -   What challenges did you face (e.g., missing data, new
        users/items)?

------------------------------------------------------------------------

## Extension (Optional)

-   Implement a **Hybrid Recommender** (combine content +
    collaborative).\
-   Use a larger dataset like
    [MovieLens](https://grouplens.org/datasets/movielens/).
