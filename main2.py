#https://open.spotify.com/playlist/1sJlI1n2fVpa9YlhBLLSzz?si=93281880e1b94467

import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from datetime import datetime
import cv2
from deepface import DeepFace
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

@app.route('/')
def start():
    return render_template('index.html')


@app.route('/link', methods= ['POST'])
def get_link():
    return str(request.form['link'])

if __name__ == '__main__':
    app.run(debug = True)


client_credentials_manager = SpotifyClientCredentials(client_id='ac1f41b0f93c45ac8af29623bdf94e0a', client_secret='f2460ff1a77e4c6db5eca633dde46d3b')
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

# playlist_link = input("Enter the link: ")
playlist_link = get_link()
print("hello")

playlist_URI = playlist_link.split("/")[-1].split("?")[0]

songs = sp.playlist_items(playlist_URI, market='IN')['items']  #will give a list of songs (meta-data)


song_name = []
song_id = []
song_popu = []
song_added_date = []
song_release_date = []
artists_col = []
for song in songs:
    song_name.append(song['track']['name'])
    song_id.append(song['track']['id'])
    song_popu.append(song['track']['popularity'])
    song_added_date.append(song['added_at'])
    song_release_date.append(song['track']['album']['release_date'])
    all_artists = song['track']['artists']
    artists = []
    for a in all_artists:
        artists.append(a['name'])
    artists_col.append(artists)

    #combining the data

df = pd.DataFrame({
    'name':song_name,
    'popularity':song_popu,
    'date_added':pd.to_datetime(song_added_date),
    'release_year':list(map(lambda x: int(x[:4]), song_release_date)),
    'artists':artists_col
    })

#audio features

features = sp.audio_features(song_id)

feat_names = list(sp.audio_features(song_id)[0].keys())

for row in range(len(features)):
    for col in range(len(feat_names)):
        df.loc[row, feat_names[col]] = features[row][feat_names[col]]
# df.head()

curr_month = datetime.today().month
curr_year = datetime.today().year

recency = list(map(lambda x: curr_month - x.month if (x.year == curr_year) else curr_month + (12 - x.month)
                    + (curr_year - x.year - 1) * 12, df['date_added']))
df['recency'] = recency

# popu ranges between 0 to 100, so normalizing it to 0 to 20
df['popularity'] = list(map(lambda x: x // 5, df['popularity']))

# deleting the rows whereever year is null
for i in range(len(df['release_year'])):
    if df.loc[i, 'release_year'] == 0:
        df.drop(i, inplace=True)


# for cosin similarity, we need the size of the vectors to be same, so we are genralizing the columns

for i in range(0,21):
    df[f"popu|{i}"] = [0] * len(df['name'])

for i in range(1980, datetime.today().year + 1):
    df[f"year|{i}"] = [0] * len(df['name'])

    # this will create dataframe with the columns of unique values in the series
df_year = pd.get_dummies(df['release_year'])
df_popu = pd.get_dummies(df['popularity'])

# assigning names to the columns
df_year.columns = map(lambda x: 'year' + '|' + str(x), df_year.columns)
df_popu.columns = map(lambda x: 'popu' + '|' + str(x), df_popu.columns)
# df_popu.head()

#now updating the columns with values wherever needed

for col in df_popu.columns:
    df[col] = df_popu[col]
for col in df_year.columns:
    df[col] = df_year[col]

# this file contains artists names which will be used for ohe
artists_excel = pd.read_excel('datasets/artists_names.xlsx')

# creating dummy dataframe for ohe-ing the artists
zeros = [0] * len(df['name'])
extra = pd.DataFrame(zeros)
for name in artists_excel['artists']:
    extra[f"artist|{name}"] = 0

new_df = pd.concat([df, extra], axis=1)
new_df.dropna(axis=0, inplace=True)

# to place 1 whenever the artist in row cell matches with the column artist
for i, row in new_df.iterrows():
    for name in row['artists']:
        if name in list(artists_excel['artists']):
            new_df.loc[i, f"artist|{name}"] = 1

new_df = new_df.drop(0, axis=1)
new_df = new_df.copy()


'''# new df for generating the recommedation vector
# we are dropping the non-integer columns as they are of no use in calulating the similarity
'''
recomm_vec_df = new_df.drop(['name', 'popularity', 'date_added', 'release_year', 'type', 'id', 'uri', 'track_href',  'analysis_url', 'artists'], axis=1)
# recomm_vec_df.columns

'''
now calculating the bias which are going to be multiplied with each of the rows individually.
for that we need to understand this that the bias must reduce the values of the older songs, so we need bias
to be between 0 and 1
1 / recency is not working as it is drastically reducing the values whih can negatively impact the recommendations
0.9 ** recency might work. For recency 3, we get the weight as 0.729, which is totally fine as we tend to listen less
to songs which are older than 3 months. Also, going below this value can trigger false recommendations
through this we are actually reducing the effect of older (added) songs
Applying only on the OHE columns as if applied on features, then avg will get affected, and we will get false predictions
'''

recomm_vec_df['bias'] = list(map(lambda x: round(0.9 ** x, 5), list(recomm_vec_df['recency'])))
for col in recomm_vec_df.columns[14:]:
    recomm_vec_df[col] = recomm_vec_df[col] * recomm_vec_df['bias']
# recomm_vec_df.head(10)

# deleting the bias and recency columns
recomm_vec_df = recomm_vec_df.dropna().drop(['bias', 'recency', 'key', 'mode', 'duration_ms', 'time_signature'], axis=1)
recomm_vec_df['tempo'] = recomm_vec_df['tempo'].apply(lambda x: (x - min(recomm_vec_df['tempo'])) / (max(recomm_vec_df['tempo'] - min(recomm_vec_df['tempo']))))
recomm_vec_df['loudness'] = recomm_vec_df['loudness'].apply(lambda x: (x - min(recomm_vec_df['loudness'])) / (max(recomm_vec_df['loudness'] - min(recomm_vec_df['loudness']))))

# this one will create the song features columns vector
recomm_vec1 = np.array(list(map(lambda col: recomm_vec_df[col].mean(), recomm_vec_df.loc[:, :"tempo"].columns)))
# this one will create the ohe columns till current year vector
recomm_vec2 = np.array(list(map(lambda col: sum(recomm_vec_df[col]), recomm_vec_df.loc[:, "popu|0":f"year|{datetime.today().year}"].columns)))
# artists only ohe columns vector
recomm_vec3 = np.array(list(map(lambda col: sum(recomm_vec_df[col]), recomm_vec_df.iloc[:, -len(artists_excel['artists']):].columns)))


#emotion code

face_cascade = cv2.CascadeClassifier('haarcascade.xml')

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()

    result = DeepFace.analyze(img_path = frame , actions=['emotion'], enforce_detection=False )

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.1,4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    emotion = result[0]['dominant_emotion']

    txt = str(emotion)

    cv2.putText(frame, txt, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break



# print(emotion)
cap.release()
cv2.destroyAllWindows()

#output: emotion


# this is the pre-processed dataset containing the 1000s of songs
data = pd.read_csv('datasets/final_data.csv')


# columns which will be used for the filtering
filt_col = ['acousticness', 'danceability', 'energy', 'loudness', 'tempo', 'valence']

# values for filtering (Emotion specific)
happy_low = [0, 0.57, 0.4, -10.4, 75 ,0.25]
sad_low = [0.2, 0.3, 0.25, -11, 70, 0]
chill_low = [0, 0.35, 0.25, -12.7, 80, 0.2]
angry_low = [0, 0.46, 0.56, -11, 90, 0.2]

happy_high = [0.75, 0.86, 1, -3, 170, 1]
sad_high = [0.9, 0.7, 0.8, -4, 160, 0.7]
chill_high = [0.85, 0.8, 0.8, -4, 165, 0.9]
angry_high = [0.6, 0.85, 1, -4, 170, 0.75]

happy_avg = [0.715, 0.7, 0.375, -6.7, 0.625, 123]
sad_avg = [0.5, 0.525, 0.55, -7.5, 0.3, 115]
chill_avg = [0.575, 0.525, 0.425, -8.35, 0.55, 122.5]
angry_avg = [0.655, 0.78, 0.3, -7.5, 0.475, 130]


i = 0
if emotion == 'happy':
    for col in filt_col:
        data = data[(data[col] > happy_low[i]) & (data[col] < happy_high[i])]
        i += 1

    sim = []
    for i in range(len(data)):
        e = data.loc[:, filt_col].iloc[i].values
        sim.append(np.linalg.norm(e - happy_avg)/70)
    data['sim'] = (np.array(sim) - max(sim)) * (-1)
    print(data['sim'])

elif emotion == 'sad':
    for col in filt_col:
        data = data[(data[col] > sad_low[i]) & (data[col] < sad_high[i])]
        i += 1
        
    sim = []
    for i in range(len(data)):
        e = data.loc[:, filt_col].iloc[i].values
        sim.append(np.linalg.norm(e - sad_avg)/70)
    data['sim'] = (np.array(sim) - max(sim)) * (-1)
    print(data['sim'])

elif emotion == 'neutral':
    for col in filt_col:
        data = data[(data[col] > chill_low[i]) & (data[col] < chill_high[i])]
        i += 1
    
    sim = []
    for i in range(len(data)):
        e = data.loc[:, filt_col].iloc[i].values
        sim.append(np.linalg.norm(e - chill_avg)/70)
    data['sim'] = (np.array(sim) - max(sim)) * (-1)
    print(data['sim'])

elif emotion == 'angry':
    for col in filt_col:
        data = data[(data[col] > angry_low[i]) & (data[col] < angry_high[i])]
        i += 1

    sim = []
    for i in range(len(data)):
        e = data.loc[:, filt_col].iloc[i].values
        sim.append(np.linalg.norm(e - angry_avg)/70)
    data['sim'] = (np.array(sim) - max(sim)) * (-1)
    print(data['sim'])


data_filtered = data.drop(['name', 'popularity', 'date_added', 'release_year', 'type', 'id', 'uri', 'track_href',  'analysis_url', 'artists', 'Unnamed: 0', 'key', 'mode', 'duration_ms', 'time_signature'], axis=1)
# data_filtered.drop(0, axis=1, inplace=True)

data_filtered['tempo'] = data_filtered['tempo'].apply(lambda x: (x - min(data_filtered['tempo'])) / (max(data_filtered['tempo'] - min(data_filtered['tempo']))))
data_filtered['loudness'] = data_filtered['loudness'].apply(lambda x: (x - min(data_filtered['loudness'])) / (max(data_filtered['loudness'] - min(data_filtered['loudness']))))
# data_filtered'

'''
Using Euclidian Distance as both magnitude and directions are important
Euclidean distance measures the distance between two points in a multidimensional space by calculating the square
root of the sum of the squared differences between their corresponding elements. It is suitable for continuous data
where the magnitude and direction of each feature are important.
'''

l1 = []
l2 = []
l3 = []
s = 0
recommendations = pd.DataFrame({'name': data['name'], 'artists':data['artists'], 'id': data['id'], 'sim': data['sim']})
for i in range(len(data_filtered)):
    # this contains the columns from start till the ohe
    data_1 = data_filtered.loc[:, :"tempo"].iloc[i].values
    # this contains the ohe columns till current year
    data_2 = data_filtered.loc[:, "popu|0":f"year|{datetime.today().year}"].iloc[i].values
    # this contains the artists only columns
    data_3 = data_filtered.iloc[:, (-len(artists_excel['artists']) - 1):-1].iloc[i].values

    sim1 = np.linalg.norm(recomm_vec1 - data_1)  # euclidian distance
    '''
    we are getting a dissimilarity score, as greater the difference 
    between the values, higher would be the score. The values which differ largerly with respect to the vector
    will tend to have a higher eucladian score
    '''

    # simply using dot product
    sim2 = np.dot(recomm_vec2, data_2)

    sim3 = np.dot(recomm_vec3, data_3)
    
    l1.append(round(sim1, 6))
    l2.append(round(sim2, 6))
    l3.append(round(sim3, 6))

l1 = (np.array(l1) - max(l1)) * (-1)  # converting it into a similarity score

# normalizing the array values to 0-1 range for proper contribution in the recommendation
l2 = (np.array(l2) - min(l2)) / (max(l2) - min(l2))
l3 = (np.array(l3) - min(l3)) / (max(l3) - min(l3)) * 0.5

score = l1 + l2 + l3
# print(type(recommendations['sim'][0]))
# print(type(recommendations['sim']), type(pd.Series(score)))
recommendations['sim'] = recommendations['sim'] + score  # as sim col is already filled with emotion effiency score


# data_filtered.iloc[:, (-len(artists_excel['artists']) - 1):-1]


recommendations.drop_duplicates(['id'], inplace=True)

# sorting the recommendations
recommendations = recommendations.sort_values(['sim'], axis=0, ascending=False)

recommendations = recommendations.reset_index().drop('index', axis=1)
# print("No. of Song Recommendations: ", len(recommendations))

