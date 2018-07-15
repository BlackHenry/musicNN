import pandas as pd
import json
import operator

top_genres = []

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
        if _ in top_genres:
            return _


def first_prepare():
    df = pd.read_csv('metadata/echonest.csv')
    genres_df = pd.read_csv('metadata/raw_tracks.csv')[['track_id', 'track_genres']]
    df = pd.merge(df, genres_df, how='inner', on=['track_id']).reset_index(drop=True)

    df.to_csv('full_db.csv')
    labels = df['track_genres']
    del df['track_genres']
    del df['track_id']
    features = df

    labels = labels.apply(func).reset_index(drop=True).dropna()
    labels = labels.apply(func2)
    genre_frequency = {}
    for genre in labels:
        for _ in genre:
            if _ in genre_frequency:
                genre_frequency[_] += 1
            else:
                genre_frequency[_] = 1
    sorted_frequency = sorted(genre_frequency.items(), key=operator.itemgetter(1))
    print(sorted_frequency[-5:])
    global top_genres
    top_genres = [_[0] for _ in sorted_frequency[-5:]]

    labels = labels.apply(func3).dropna()
    features = features.loc[labels.index].reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    #labels = pd.DataFrame.from_items(zip(labels.index, labels.values)).T
    labels = pd.get_dummies(labels)
    labels.to_csv('labels2.csv')
    features.to_csv('features2.csv')


def normalize():
    df = pd.read_csv('features.csv')
    df['tempo'] /= max(df['tempo'])
    df.to_csv('features.csv')


first_prepare()







