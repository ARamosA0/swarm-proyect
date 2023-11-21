from flask import Flask, render_template, request, make_response, g
from redis import Redis
import os
import socket
import random
import json
import logging
import pandas as pd
import numpy as np
import math

option_a = os.getenv('OPTION_A', "Cats")
option_b = os.getenv('OPTION_B', "Dogs")
hostname = socket.gethostname()

app = Flask(__name__)

gunicorn_error_logger = logging.getLogger('gunicorn.error')
app.logger.handlers.extend(gunicorn_error_logger.handlers)
app.logger.setLevel(logging.INFO)

def get_redis():
    if not hasattr(g, 'redis'):
        g.redis = Redis(host="redis", db=0, socket_timeout=5)
    return g.redis



#Convierte la data a binarios usando Numpy
def movie_lens_to_binary(input_file, output_file):
    # Load MovieLens data using Pandas
    ratings = pd.read_csv(input_file, sep='\t', header=None,
                          names=['userId', 'movieId', 'rating', 'rating_timestamp'])
    # Convert to NumPy array
    np_data = np.array(ratings[['userId', 'movieId', 'rating']])
    # Write to binary file
    with open(output_file, "wb") as bin_file:
        bin_file.write(np_data.astype(np.int32).tobytes())


def binary_to_pandas_with_stats(bin_file, num_rows=10):
    # Read binary data into NumPy array
    with open(bin_file, 'rb') as f:
        binary_data = f.read()
    # Convert binary data back to NumPy array
    np_data = np.frombuffer(binary_data, dtype=np.int32).reshape(-1, 3)  # Assuming 3 columns
    # Convert NumPy array to Pandas DataFrame
    df = pd.DataFrame(np_data, columns=['userId', 'movieId', 'rating'])
    return df

def consolidate_data(df):
    # Group by 'userId' and 'movieId' and calculate the mean of 'rating'
    consolidated_df = df.groupby(['userId', 'movieId'])['rating'].mean().unstack()
    return consolidated_df


def get_data():
    # Toget the data:
    # 100K --> wget https://files.grouplens.org/datasets/movielens/ml-100k.zip 
    # 1M --> wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
    # 10M --> wget https://files.grouplens.org/datasets/movielens/ml-10m.zip
    movie_lens_to_binary('/usr/local/app/data/ratings.dat', 'output_binary.bin')
    data = binary_to_pandas_with_stats('output_binary.bin', num_rows=10)
    print(data)
    #df = pd.read_csv('ratings.data')
    consolidated_df = consolidate_data(df)
    return consolidated_df


def new_manhattan(rating1, rating2):
    distance = 0
    commonRatings = False

    for key in rating1:
        if key in rating2:
            if not math.isnan(rating1[key]) and not math.isnan(rating2[key]):
                distance += abs(rating1[key] - rating2[key])
                commonRatings = True
            else:
                pass 

    if commonRatings:
        return distance
    else:
        return -1

def computeNearestNeighbor(username, users):
    """creates a sorted list of users based on their distance to username"""
    distances = []
    for user in users:
        if user != username:

          #Cambiar por la distancia a usar
            #distance = similitud_coseno_nan(users[user], users[username])
            #distance = correlacion_pearson_nan(users[user], users[username])
            distance = new_manhattan(users[user], users[username])
            distances.append((distance, user))
    # sort based on distance -- closest first
    distances.sort()
    return distances

def recommend(username, users):
    """Give list of recommendations"""
    # first find all nearest neighbors
    nearest_neighbors = computeNearestNeighbor(username, users)

    recommendations = []

    # now find bands neighbors rated that user didn't
    for nearest_neighbor in nearest_neighbors[:5]:
        neighbor_ratings = users[nearest_neighbor[1]]
        user_ratings = users[username]

        neighbor_ratings = {artist: rating for artist, rating in neighbor_ratings.items() if not math.isnan(rating)}
        user_ratings = {artist: rating for artist, rating in user_ratings.items() if not math.isnan(rating)}

        for artist in neighbor_ratings:
            if artist not in user_ratings:
                recommendations.append((artist, neighbor_ratings[artist]))
    # using the fn sorted for variety - sort is more efficient
    return sorted(recommendations, key=lambda artist_tuple: artist_tuple[1], reverse=True)


def limpia(np1, np2):
    mask = ~np.isnan(np2)
    np1 = np1[mask]
    np2 = np2[mask]

    np1, np2 = np2, np1

    mask = ~np.isnan(np2)
    np1 = np1[mask]
    np2 = np2[mask]

    np1, np2 = np2, np1

    return pd.DataFrame({'A': np1, 'B': np2})

def computeManhattanDistance(ax, bx):
    return np.sum(np.abs(ax - bx))

def computeNearestNeighbor(username, users_df):
    user_data = np.array(users_df.loc[username])
    distances = np.empty((users_df.shape[0],), dtype=float)

    for i, (index, row) in enumerate(users_df.iterrows()):
        if index != username:
            ax = np.array(row)
            bx = np.array(user_data)
            temp = limpia(ax, bx)
            ax = np.array(temp["A"].tolist())
            bx = np.array(temp["B"].tolist())
            distances[i] = computeManhattanDistance(ax, bx)

    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]

    return list(zip(sorted_distances, users_df.index[sorted_indices]))

def recommend(username, users_df):
    nearest_neighbors = computeNearestNeighbor(username, users_df)
    user_data = np.array(users_df.loc[username])
    user_items = np.isnan(user_data)

    # Inicializar las recomendaciones como un diccionario vacío
    recommendations = {}

    # Iterar sobre los vecinos más cercanos
    for distance, neighbor in nearest_neighbors:
        neighbor_data = np.array(users_df.loc[neighbor])
        neighbor_items = np.isnan(neighbor_data)

        # Encontrar ítems que el vecino haya valorado y el usuario no
        new_items = np.logical_and(neighbor_items, ~user_items)

        # Actualizar las recomendaciones con la puntuación ponderada por la distancia
        for item_index, has_new_item in enumerate(new_items):
            if has_new_item:
                if item_index not in recommendations:
                    recommendations[item_index] = 0
                recommendations[item_index] += neighbor_data[item_index] / (distance + 1)

    # Ordenar las recomendaciones de mayor a menor puntuación
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

    # Obtener los índices de los elementos recomendados
    recommended_items = [item_index for item_index, score in sorted_recommendations]

    # Devolver las recomendaciones
    return recommended_items

@app.route("/", methods=['POST','GET'])
def hello():
    print("***************************************************************************")
    print("INICIA FUNCION")
    data = get_data()
    
    voter_id = request.cookies.get('voter_id')
    if not voter_id:
        voter_id = hex(random.getrandbits(64))[2:-1]
    if request.method == 'POST':
        id = request.form["idusuario"]
        #distance = recommend(id,data)
        print("------------------------------------------------------")
        print("CON 100k")
        #nn = computeNearestNeighbor(int(id), data)
        recom = recommend(int(id), data)
        print(recom, "RECOM")
        print("------------------------------------------------------")
        redis = get_redis()
        #name, value = distance[0]
        #data_json = json.dumps({'voter_id':voter_id, "name":name, "value":value})
        data_json = json.dumps({'voter_id':voter_id,"value":recom[-1]})
        redis.rpush('recommend', data_json)
        print("--------------------------------------------------------------------")
        print("Se enviaron los datos a REDIS")
        #print(data_json)
    print("***************************************************************************")
    resp = make_response(render_template('index.html'))
    return resp


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True, threaded=True)
