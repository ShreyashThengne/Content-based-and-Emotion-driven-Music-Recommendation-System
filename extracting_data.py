
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import datetime
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv

load_dotenv()

client_credentials_manager = SpotifyClientCredentials(client_id=os.getenv('CLIENT_ID'), client_secret=os.getenv('CLIENT_SECRET'))

sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


# playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DX0XUfTFmNBRM?si=7934cecad3f3439c"
playlist_link = input("Enter the link: ")

playlist_URI = playlist_link.split("/")[-1].split("?")[0]

songs = sp.playlist_tracks(playlist_URI, market='IN')['items']  #will give a list of songs (meta-data)

song_name = []
song_id = []
song_popu = []
song_added_date = []
song_release_date = []
for song in songs:
    song_name.append(song['track']['name'])
    song_id.append(song['track']['id'])
    song_popu.append(song['track']['popularity'])
    song_added_date.append(song['added_at'])
    song_release_date.append(song['track']['album']['release_date'])
 
release_year = map(lambda x: int(x[:4]), song_release_date)

import pandas as pd

df = pd.DataFrame({
    'name':song_name,
    'popularity':song_popu,
    'date_added':song_added_date,
    'release_year':release_year
    })

df['name'].replace("", np.nan, inplace=True)
df.dropna(subset = ['name'], inplace=True)

features = sp.audio_features(song_id)

feat_names = list(sp.audio_features(song_id)[0].keys())

for row in range(len(features)):
    for col in range(len(feat_names)):
        df.loc[row, feat_names[col]] = features[row][feat_names[col]]


artists_col = []
for i in range(len(songs)):
    all_artists = songs[i]['track']['artists']
    artists = []
    for a in all_artists:
        artists.append(a['name'])
    artists_col.append(artists)
df['artists'] = artists_col

df.dropna(subset=['name'], inplace=True)

df['popularity'] = df['popularity'].astype(int)
df['release_year'] = df['release_year'].astype(int)

l = df

df['popularity'] = list(map(lambda x: x // 5, df['popularity']))

for i in range(0,21):
    df[f"popu|{int(i)}"] = [int(0)] * len(df['name'])

for i in range(1980, datetime.date.today().year + 1):
    df[f"year|{int(i)}"] = [int(0)] * len(df['name'])

df_year = pd.get_dummies(df['release_year'].astype(int), prefix="year", prefix_sep='|').astype(int)
df_popu = pd.get_dummies(df['popularity'].astype(int), prefix="popu", prefix_sep='|').astype(int)

for col in df_popu.columns:
    df[col] = df_popu[col]
for col in df_year.columns:
    df[col] = df_year[col]

df.columns[:]

new_df = df.copy()

artists_excel = pd.read_excel('datasets/artists_names.xlsx')

excel_df = pd.read_csv("datasets/final_data.csv")

art = []
for i in range(len(excel_df['name'])):
    b = 0
    ele = ''
    new_l = []
    for ch in excel_df.loc[i, 'artists']:
        if b == 1:
            if ch == "'":
                b = 0
                new_l.append(ele)
                ele = ''
            else:
                ele += ch
        elif ch == "'":
            b = 1

    art.append(new_l)

excel_df['artists'] = art



zeros = [0] * len(new_df['name'])
extra = pd.DataFrame(zeros)

for name in artists_excel['artists']:
    extra[f"artist|{name}"] = 0
    
new_df = pd.concat([new_df, extra], axis=1)

excel_df = pd.concat([new_df, excel_df])

print(len(excel_df), len(df))

excel_df = pd.concat([excel_df, new_df])
for name in artists_excel['artists']:
    excel_df[f"artist|{name}"] = 0

excel_df.drop(columns = ['Unnamed: 0', '0', 'year|0', 'year|1960', 'year|1964', 0], axis=1, inplace=True, errors='ignore')


excel_df = excel_df.reset_index(drop=True)

excel_df.dropna(subset=['name'], inplace=True)

excel_df.reset_index(drop=True, inplace=True)
for i in range(len(excel_df['name'])):
    for name in np.array(excel_df.loc[i, 'artists']):
        if name in np.array(artists_excel['artists']):
            excel_df.loc[i, f"artist|{name}"] = 1

excel_df = excel_df.drop_duplicates('id')
excel_df.columns


len(excel_df)

excel_df = excel_df.reset_index().drop('index', axis=1)
excel_df.columns

excel_df.drop(columns = ['Unnamed: 0.2','Unnamed: 0.1','Unnamed: 0', '0', 'year|0', 'year|1960', 'year|1964', 0], axis=1, inplace=True, errors='ignore')

feat_names = ["danceability", "energy",	"key",	"loudness",	"mode",	"speechiness",	"acousticness",	"instrumentalness",	"liveness",	"valence",	"tempo"]

model = load_model('model.h5')

x = excel_df[['acousticness', 'danceability', 'energy', 'loudness', 'tempo', 'valence', 'speechiness']]
x.columns

from joblib import load
ss = load('scaler.joblib')
X_scaled = ss.transform(x)

y=model.predict(X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1]))
len(y)

# ['angry', 'chill', 'happy', 'sad']
emo = pd.Series(np.argmax(y.reshape(y.shape[0], y.shape[2]), axis=1)).map({0:'angry', 1:'chill', 2:'happy', 3:'sad'})

x['emotion'] = emo

excel_df['emotion'] = emo
excel_df.head()

excel_df.to_csv('datasets/final_data.csv')

print("\nSong list updated!")