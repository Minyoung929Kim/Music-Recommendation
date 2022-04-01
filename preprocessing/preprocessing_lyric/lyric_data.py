from random import seed
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius as lg


class LyricDataCollector:

    def __init__(self,
                 spotify_client_id=None,
                 spotify_client_secret=None,
                 genius_token=None):
        self.spotify_client_id = spotify_client_id
        self.spotify_secret = spotify_client_secret
        self.genius_token = genius_token

        self.genius = lg.Genius(genius_token,
                                skip_non_songs=True,
                                excluded_terms=["(Remix)", "(Live)"],
                                remove_section_headers=True,
                                timeout=20)
        self.genius.verbose = False
        self.client_credentials_manager = SpotifyClientCredentials(
            client_id=spotify_client_id, client_secret=spotify_client_secret)
        self.sp = spotipy.Spotify(
            client_credentials_manager=self.client_credentials_manager)

    @classmethod
    def from_secret(cls, secret_path):
        kwargs = {}
        with open(secret_path, 'r') as f:
            for line in f:
                name, token = line.strip().split('=')
                kwargs[name] = token
        return cls(**kwargs)

    def get_genre_seeds(self):
        return self.sp.recommendation_genre_seeds()['genres']

    def get_recommendations(self, seed_genres, limit=20, country='US'):
        return self.sp.recommendations(seed_genres=seed_genres,
                                       limit=limit,
                                       country=country)

    def get_track_info(self, track):
        track_info = self.sp.audio_features(tracks=[track['uri']])[0]
        if track_info is None:
            return None

        for key, value in track.items():
            if isinstance(value, (dict, list)):
                continue
            track_info[key] = value
        info_df = pd.DataFrame([track_info])
        return info_df

    def scrape_lyrics(self, artistname, songname):
        try:
            song = self.genius.search_song(title=songname,
                                           artist=artistname,
                                           get_full_info=True)
            return song.lyrics
        except:
            return None

    def lyrics_onto_frame(self, df, artist_name):
        for i, x in enumerate(df['name']):
            test = self.scrape_lyrics(artist_name, x)
            df.loc[i, 'lyrics'] = test
        return df

    def get_songs(self, seed_genres=None, recommendations=None):
        if recommendations is None:
            if seed_genres is None:
                genre_seeds = self.get_genre_seeds()[:5]
            recommendations = self.get_recommendations(genre_seeds)
        if isinstance(recommendations['tracks'], dict):  # from a playlist
            recommendations['tracks'] = [
                x['track'] for x in recommendations['tracks']['items']
            ]

        lyrics_df = []
        for track in recommendations['tracks']:
            track_metadata = self.get_track_info(track)
            if track_metadata is None:
                continue
            track_df = self.lyrics_onto_frame(track_metadata, track['artists'][0]['name'])
            lyrics_df.append(track_df)

        final_df = pd.concat(lyrics_df).dropna()

        return final_df

    def set_verbose(self, verbose=False):
        self.genius.verbose = verbose
