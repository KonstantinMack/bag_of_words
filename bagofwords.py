import pandas as pd
import numpy as np
import re
import operator
from collections import Counter
from nltk.stem import PorterStemmer


def make_counters(picklelist):
    """
    Takes a list of pickle objects and returns two objects:
        counter_all: Counter of all words of all artists together
        counter_artists: a list where each element is a list of Counters
                        for each song of one artist
    Uses stemming during the tokenization
    """
    lyrics = []
    lyrics_all = ""
    for i in picklelist:
        songs = pd.read_pickle("lyrics/" + i)
        lyrics.append(songs)
        for song in songs:
            lyrics_all += song
    ps = PorterStemmer()
    counter_all = Counter([ps.stem(i) for i in lyrics_all.split()])
    counter_artists = []
    for liste in lyrics:
        counter_artists.append([Counter([ps.stem(j) for j in i.split()]) for i in liste])
    return counter_all, counter_artists


def make_df(counter_all, counter_artists, artist_list):
    '''
    Takes counter_all and counter_artists (from make_counters()) and
    creates a dataframe with artist songs as rows and words as columns.
    Artist_list must contain names for the artists in the same order
    as in the pickle_list.
    '''
    rows = []
    for i, j in enumerate(counter_artists):
        rows += [artist_list[i] for k in j]
    data = []
    for artist in counter_artists:
        for song in artist:
            row = []
            for word in counter_all:
                row.append(song[word])
            data.append(row)
    df = pd.DataFrame(data, columns=counter_all.keys(), index=rows)
    return df


def tf_idf(df):
    """
    Calculates TF-IDF for a matrix
    """
    notnull = df > 0
    docs_per_word = notnull.sum()
    idf = np.log(((1 + df.shape[0])/(1 + docs_per_word))) +1
    tf_idf = df * idf
    sq = tf_idf ** 2
    sum_sq = np.sqrt(sq.sum(axis=1))
    tf_idf = (tf_idf.T / sum_sq).T
    return tf_idf


# Implementation of Bayes' Rule
def prob_word_art(df, word, artist, k):
    """
    Helper function for Bayes' Rule
    """
    return (sum(df.loc[artist][word]) + k) / (sum(df[word]) + 2*k)


def prob_query_art(df, query, artist, k):
    """
    Helper function for Bayes' Rule
    """
    query = re.sub("\<i\>.{1,50}\<\/i\>|\<br\>|\n|\<\/div\>|\(|\)|-|\.|\;|\:|\!|\,|'|\?|\&quot", " ", query)
    query = query.lower().split()
    ps = PorterStemmer()
    query = [ps.stem(i) for i in query]
    summe = 0
    for i in query:
        if i in df.columns:
            summe += np.log(prob_word_art(df, i, artist, k))
    return np.exp(summe)


def prob_art(df, artist):
    """
    Helper function for Bayes' Rule
    """
    pro_art = df.loc[artist].sum().sum()
    con_art = df.loc[~df.index.isin([artist])].sum().sum()
    art_prob = pro_art / (pro_art + con_art)
    return art_prob


def prob_bayes(df, query, artist_list, k):
    """
    Calculates probabilities according to Bayes' Rule
    """
    p_query_art_list = []
    p_art_prob_list = []
    for artist in artist_list:
        p_query_art_list.append(prob_query_art(df, query, artist, k))
        p_art_prob_list.append(prob_art(df, artist))
    a_list = np.array(p_query_art_list)
    b_list = np.array(p_art_prob_list)
    final_probs = []
    for idx, artist in enumerate(artist_list):
        zaehler = a_list[idx] * b_list[idx]
        nenner = zaehler + sum(a_list[np.arange(len(a_list))!=idx] * b_list[np.arange(len(b_list))!=idx])
        final_probs.append(zaehler/nenner)
    return final_probs

# Main program:
def main(pickle_list, artist_list, query, k=0.05):
    counter_all, counter_artists = make_counters(pickle_list)
    df = make_df(counter_all, counter_artists, artist_list)
    df_tfidf = tf_idf(df)
    probs = prob_bayes(df_tfidf, query, artist_list, k)
    probs_dict = dict(zip(artist_list, probs))
    sorted_probs = sorted(probs_dict.items(), key=operator.itemgetter(1), reverse=True)
    print("Query: ", query)
    print("Probabilities:")
    for i in sorted_probs:
        print(i[0], ": ", "{:04.2f} %".format(i[1]*100))


main(["eminem.pkl", "madonna.pkl", "nas.pkl", "bep.pkl", "spears100.pkl", "justinbieber.pkl"], ["Eminem", "Madonna", "Nas", "BEP", "Britney", "Bieber"], "bitch")
