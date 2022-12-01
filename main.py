import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

game_reviews_df = pd.read_csv('steam_reviews.csv')
# print(game_reviews_df)

clean_game_reviews_df = game_reviews_df.drop(['review', 'is_early_access_review'], axis=1)


clean_game_reviews_df.replace({"Not Recommended": 0, "Recommended": 1}, inplace=True)
# print(clean_game_reviews_df)
matrix = clean_game_reviews_df.pivot_table(columns='title', index='date_posted', values='recommendation')
print(matrix)