import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

game_reviews_df = pd.read_csv('steam-200k.csv')

clean_game_reviews_df = game_reviews_df.drop(['0', 'purchase'], axis=1)
clean_game_reviews_df = clean_game_reviews_df.rename({'151603712': 'User', 'The Elder Scrolls V Skyrim': 'Game', '1.0': 'PlayTime'}, axis=1)

# hacemos una matriz con los juegos como columnas, los usuarios como indices y los valores entre ambos será el tiempo de juego
# también llenamos los valores nulos con 0
matrix = clean_game_reviews_df.pivot_table(columns='Game', index='User', values='PlayTime', fill_value=0)

#funcion para centrar una matriz
def center(row):
    new_row = (row - row.mean()) / (row.max() - row.min())
    return new_row

#centramos la matriz para normalizar los datos 
matrix_std = matrix.apply(center)

def gameRec(game):
    centered_matrix_values = matrix_std[game]
    #Calcula la correlación de pearson de un juego con los demás
    centered_matrix_values = matrix.corrwith(centered_matrix_values).dropna()
    #crea un dataframe que muestra las veces que cada juego ha sido jugado y la media del tiempo que ha sido jugado
    gameData = clean_game_reviews_df.groupby('Game').agg({'PlayTime': [np.size, np.mean]})
    #filtro que elimina los juegos jugados por menos de 100 usuarios.
    gameSim = gameData['PlayTime']['size'] >= 100
    #juntamos la columna de similaridad del dataframe filtrado con el dataframe que tiene el resto de datos
    df = gameData[gameSim].join(pd.DataFrame(centered_matrix_values, columns=['similarity']), ['Game'])
    return df.sort_values(['similarity'], ascending=False)[:6]

print(gameRec("Borderlands 2"))
