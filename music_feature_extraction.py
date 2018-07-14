import spotipy
from spotipy import util
import pandas as pd
import numpy as np
from keras import models


token = util.prompt_for_user_token(username='blackhenri715@gmail.com', scope='user-read-private', client_id='03d87d8755164fefa64eaad1e77031d1', client_secret='ee60156fe7e34ce5891d20a33c78fb59', redirect_uri='http://localhost/')
sp = spotipy.Spotify(auth=token)

track_features = sp.audio_features('https://open.spotify.com/track/0PHwCbA310LLIMcSDpQbgF')[0]
features = [track_features['acousticness'], track_features['danceability'], track_features['energy'],
            track_features['instrumentalness'], track_features['liveness'], track_features['speechiness'],
            track_features['tempo']/251.072, track_features['valence']]
df = pd.DataFrame(pd.Series(features)).T

model = models.load_model('model.h5')
prediction = model.predict(df)
predicted_genre = int(np.argmax(prediction))
genre_ids = ['10', '12', '15', '17', '18', '2', '21', '240', '25', '26', '286', '3', '31',
             '33', '360', '4', '5', '514', '538', '58', '66', '8', '89']
genre_data = pd.read_csv('metadata/raw_genres.csv')
genre_name = genre_data[genre_data['genre_id'] == int(genre_ids[predicted_genre])]['genre_title'].values[0]
print(genre_name, prediction)

