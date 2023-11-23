from flask import Flask, render_template, request, make_response, g
from redis import Redis
import os
import socket
import random
import json
import logging
import math
import pandas as pd
import numpy as np

app = Flask(__name__)

gunicorn_error_logger = logging.getLogger('gunicorn.error')
app.logger.handlers.extend(gunicorn_error_logger.handlers)
app.logger.setLevel(logging.INFO)

def get_redis():
    if not hasattr(g, 'redis'):
        g.redis = Redis(host="redis", db=0, socket_timeout=5)
    return g.redis

ratings = pd.read_csv('ratings.csv')

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

def cosine_similarity_manual(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0  # Evitar la división por cero
    
    return dot_product / (magnitude_vec1 * magnitude_vec2)

def find_nearest_neighbors_manual(user_id, ratings_df, num_neighbors=10):
    user_ratings = ratings_df.loc[ratings_df['userId'] == user_id].drop(columns=['userId', 'timestamp']).values[0]
    other_users_ratings = ratings_df.loc[ratings_df['userId'] != user_id].drop(columns=['userId', 'timestamp']).values

    similarities = np.array([
        cosine_similarity_manual(user_ratings, row)
        for row in other_users_ratings
    ])

    nearest_neighbors_indices = np.argsort(similarities)[::-1][:num_neighbors]
    nearest_neighbors_ids = ratings_df.iloc[nearest_neighbors_indices]['userId'].tolist()

    return nearest_neighbors_ids

def manhattan(rating1, rating2):
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
        #print(user)
        if user != username:

          #Cambiar por la distancia a usar
            #distance = similitud_coseno_nan(users[user], users[username])
            #distance = correlacion_pearson_nan(users[user], users[username])
            distance = manhattan(users[user], users[username])
            distances.append((distance, user))
    # sort based on distance -- closest first
    distances.sort()
    return distances

def recommend(username, users):
    """Give list of recommendations"""
    # first find all nearest neighbors
    nearest_neighbors = computeNearestNeighbor(username, users)
    #print(nearest_neighbors)

    recommendations = []

    # now find bands neighbors rated that user didn't
    for nearest_neighbor in nearest_neighbors[:5]:
        print(nearest_neighbor)
        neighbor_ratings = users[nearest_neighbor[1]]
        user_ratings = users[username]
        #print(user_ratings)

        neighbor_ratings = {artist: rating for artist, rating in neighbor_ratings.items() if not math.isnan(rating)}
        user_ratings = {artist: rating for artist, rating in user_ratings.items() if not math.isnan(rating)}

        #print(neighbor_ratings)
        #print(user_ratings)
        for artist in neighbor_ratings:
            if artist not in user_ratings:
                recommendations.append((artist, neighbor_ratings[artist]))
                #print(neighbor_ratings[artist])

    # using the fn sorted for variety - sort is more efficient
    return sorted(recommendations, key=lambda artist_tuple: artist_tuple[1], reverse=True)

@app.route("/", methods=['POST', 'GET'])
def hello():
    voter_id = request.cookies.get('voter_id')
    if not voter_id:
        voter_id = hex(random.getrandbits(64))[2:-1]

    if request.method == 'POST':
        redis = get_redis()
        
        if 'calculate' in request.form:
            user_id = int(request.form.get('user_id'))
            #user_name = request.form.get('user_id')
            
            cn = find_nearest_neighbors_manual(user_id, ratings)
            #nn = computeNearestNeighbor(user_id, ratings)
            #r = recommend(user_name, ratings)
            # Convertir la lista a formato JSON antes de almacenar en Redis
            neighbors_data = json.dumps({'user_id': user_id, 'neighbors': cn})
            redis.rpush('cosine_neighbors', neighbors_data)
            app.logger.info(neighbors_data)
            if redis.exists('cosine_neighbors'):
                app.logger.info('Data uploaded to Redis successfully')
            else:
                app.logger.error('Failed to upload data to Redis')

    resp = make_response(render_template(
        'index.html',
        option_a=os.getenv('OPTION_A', "Cats"),
        option_b=os.getenv('OPTION_B', "Dogs"),
        hostname=socket.gethostname(),
        similarity=None,
        ratings_data=None,
    ))
    resp.set_cookie('voter_id', voter_id)
    return resp

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True, threaded=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True, threaded=True)
