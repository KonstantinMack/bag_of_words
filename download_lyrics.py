import requests
import re
import random
import time
import pickle

"""
Script to download lyrics of an artist from azlyrics.com
Use 'download_main' function and pass it the artist and the number of songs to download.
Note: artist has to be passed in the same format as it appears in the URL when looking it up on azlyrics.com directly.
"""


def download_songlist(artist):
    """
    returns a list of all songs of that artist
    """
    url = "https://www.azlyrics.com/" + artist[0] + "/" + artist + ".html"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    seite = requests.get(url, headers=headers)
    expression = "lyrics/(" + artist + "/\w+.html)"
    songliste = list(set(re.findall(expression, seite.text)))
    return songliste


def download_urls(songliste):
    """
    extracts the URLs for each song from the songlist
    """
    song_url = []
    for i in songliste:
            url = "https://www.azlyrics.com/lyrics/" + i
            song_url.append(url)
    return song_url


def download_song(url):
    """
    takes a URL and downloads the lyrics from that URL
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    seite = requests.get(url, headers=headers)
    text = re.findall("Sorry about that. -->(.+)<!-- MxM banner -->", seite.text, re.DOTALL)[0]
    text = re.sub("\<i\>.{1,50}\<\/i\>|\<br\>|\n|\r|\<\/div\>|\(|\)|-|\.|\;|\:|\!|\,|'|\?|\&quot", " ", text)
    text = text.lower()
    return text


def download_songtexts(song_url, n):
    """
    goes through a list of URL and downloads the lyrics from each URL
    """
    songs = []
    if n is None:
        for i, j in enumerate(song_url, 1):
            songs.append(download_song(j))
            print(f"song {i} of {len(song_url)} done...")
            time.sleep(10)
    else:
        for i, j in enumerate(random.sample(song_url, n), 1):
            songs.append(download_song(j))
            print(f"song {i} of {n} done...")
            time.sleep(10)
    return songs


def download_main(artist, n=None):
    """
    Main function: enter artist and number of songs to download
    If n = None, then all available songs of that artist will be parsed
    Note: artist has to be passed in the same format as it appears in the URL when looking it up on azlyrics.com directly

    Returns a pickle file containing all the lyrics
    """
    songlist = download_songlist(artist)
    urls = download_urls(songlist)
    lyrics = download_songtexts(urls, n)
    with open(artist + str(len(lyrics)) + '.pkl', 'wb') as f:
        pickle.dump(lyrics, f)


download_main("justinbieber", 75)
