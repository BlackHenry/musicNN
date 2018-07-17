import pandas as pd
import json
import operator

tracks_per_genre = 600
genre_ids = [4, 5, 10, 12, 15]
used_genre_slots = {4: 0, 5: 0, 10: 0, 12: 0, 15: 0}


def func(x):
    if x[0] == '[':
        return json.loads(x.replace('\'', '\"'))


def func2(x):
    new_x = []
    for _ in x:
        _ = str(_['genre_id'])
        new_x.append(_)
    return new_x


def func3(x):
    for _ in x:
        _ = int(_)
        if _ == 8:
            _ = 5
        if _ in genre_ids and used_genre_slots[_] < tracks_per_genre:
            used_genre_slots[_] += 1
            return _


def prepare_data():
    df = pd.read_csv('metadata/ground_truth.csv')
    genres_df = pd.read_csv('metadata/raw_tracks.csv')[['track_id', 'track_genres']]
    df = pd.merge(df, genres_df, how='inner', on=['track_id']).reset_index(drop=True)

    df.to_csv('full_db.csv')
    labels = df['track_genres']
    del df['track_genres']
    del df['track_id']
    features = df

    labels = labels.apply(func)
    labels = labels.apply(func2)
    labels = labels.apply(func3).dropna()
    features = features.loc[labels.index].reset_index(drop=True)
    features['tempo'] /= max(features['tempo'])
    labels = labels.reset_index(drop=True)
    labels = pd.get_dummies(labels)
    labels.to_csv('labels2.csv')
    features.to_csv('features2.csv')


def normalize():
    df = pd.read_csv('features.csv')
    df['tempo'] /= max(df['tempo'])
    df.to_csv('features.csv')


prepare_data()


