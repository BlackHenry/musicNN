import pandas as pd
import json


def func(x):
    if x[0] == '[':
        return json.loads(x.replace('\'', '\"'))


def func2(x):
    new_x = []
    for _ in x:
        _ = str(_['genre_id'])
        new_x.append(_)
    return new_x[0]


def first_prepare():
    df = pd.read_csv('metadata/echonest.csv')
    genres_df = pd.read_csv('metadata/raw_tracks.csv')[['track_id', 'track_genres']]
    df = pd.merge(df, genres_df, how='inner', on=['track_id']).reset_index(drop=True)

    labels = df['track_genres']
    del df['track_genres']
    del df['track_id']
    features = df

    labels = labels.apply(func).reset_index(drop=True).dropna()
    features = features.loc[labels.index].reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    labels = labels.apply(func2)
    #labels = pd.DataFrame.from_items(zip(labels.index, labels.values)).T
    labels = pd.get_dummies(labels)
    labels.to_csv('labels2.csv')
    features.to_csv('features2.csv')


def normalize():
    df = pd.read_csv('features.csv')
    df['tempo'] /= max(df['tempo'])
    df.to_csv('features.csv')


first_prepare()
df = pd.read_csv('labels2.csv')
a = df.columns
print(a)







