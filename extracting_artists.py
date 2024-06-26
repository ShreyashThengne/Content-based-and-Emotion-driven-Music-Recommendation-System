
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

client_credentials_manager = SpotifyClientCredentials(client_id=os.getenv('CLIENT_ID'), client_secret=os.getenv('CLIENT_SECRET'))

sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


# playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DX0XUfTFmNBRM?si=7934cecad3f3439c"
playlist_link = input("Enter the link: ")

playlist_URI = playlist_link.split("/")[-1].split("?")[0]

songs = sp.playlist_tracks(playlist_URI, market='IN')['items']  #will give a list of songs (meta-data)

df = pd.DataFrame([])
artists_col = []
for i in range(len(songs)):
    all_artists = songs[i]['track']['artists']
    artists = []
    for a in all_artists:
        artists.append(a['name'])
    artists_col.append(artists)
df['artists'] = artists_col

df = df.explode('artists').drop_duplicates().reset_index().drop('index', axis=1)
df1 = pd.read_excel("datasets/artists_names.xlsx")
df1.drop('Unnamed: 0', axis=1, inplace=True)

df1 = pd.concat([df1, df]).drop_duplicates().reset_index().drop('index', axis=1)

df1.to_excel('datasets/artists_names.xlsx')

print("\nArtist list updated!")