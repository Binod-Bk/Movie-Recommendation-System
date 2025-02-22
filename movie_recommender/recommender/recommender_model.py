import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

# Get the absolute path to the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model using joblib
model_path = os.path.join(BASE_DIR, "movie_recommender.joblib")

if not os.path.exists(model_path):
    print("⚠️ Error: Model file does not exist at", model_path)
else:
    print("✅ Model file found at", model_path)

try:
    latent_matrix_1_df, latent_matrix_2_df = joblib.load(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("🚨 Error loading model:", e)


def recommend_movie(movie_name):
    """
    Recommend similar movies based on hybrid collaborative and content filtering.
    :param movie_name: str, Movie title to get recommendations for.
    :return: list, Top 10 recommended movies.
    """
    if movie_name not in latent_matrix_1_df.index:
        return ["Movie not found"]

    # Get the feature vectors of the selected movie
    a_1 = np.array(latent_matrix_1_df.loc[movie_name]).reshape(1, -1)
    a_2 = np.array(latent_matrix_2_df.loc[movie_name]).reshape(1, -1)

    # Compute cosine similarity with other movies
    score_1 = cosine_similarity(latent_matrix_1_df, a_1).reshape(-1)
    score_2 = cosine_similarity(latent_matrix_2_df, a_2).reshape(-1)

    # Compute the hybrid similarity score
    hybrid = (score_1 + score_2) / 2.0

    # Create a dataframe of similarities
    similar = pd.DataFrame({"hybrid": hybrid}, index=latent_matrix_2_df.index)

    # Sort by hybrid similarity score (descending)
    similar.sort_values("hybrid", ascending=False, inplace=True)

    # Return the top 10 recommended movies
    return similar.index[1:11].tolist()  # Exclude the input movie itself
