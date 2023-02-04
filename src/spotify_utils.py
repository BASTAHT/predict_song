import pandas as pd
import spotipy

from . import parameters as p


def auth_spotify(
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    username: str,
) -> spotipy.client.Spotify:
    """
    Method to authenticate the spotify connection

    Args:
        client_id: The client id
        client_secret: the client secret
        redirect_uri: the redirect uri
        username: the username

    Returns:
        an authenticated API connection
    """
    scope = "user-library-read user-library-modify playlist-modify-private"
    auth_manager = spotipy.SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        username=username,
        scope=scope,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def get_playlist_data(
    connection: spotipy.client.Spotify, playlist_id: str, liked: bool
) -> pd.DataFrame:
    """
    Get data for all songs in a playlist

    Args:
        connection: the authenticated API connection
        playlist_id: The ID of the playlist
        liked: Marker if the songs in the playlist are liked or not

    Returns:
        dataframe containing the data of all songs in the playlist
    """
    length = connection.playlist(playlist_id)["tracks"]["total"]
    df = pd.DataFrame
    for i in range(0, length, 100):
        playlist = connection.playlist_tracks(
            playlist_id, fields="items.track", offset=i
        )
        artist = []
        song = []
        ids = []
        for track in playlist["items"]:
            try:
                nummer = track["track"]
                artist.append(nummer["artists"][0]["name"])
                song.append(nummer["name"])
                ids.append(nummer["id"])
            except:
                print(f"song {track} can not be found")
        songdata = connection.audio_features(ids)
        songdata = filter(None, songdata)
        df_data = pd.DataFrame(songdata).set_index("id")
        df_info = pd.DataFrame(
            list(zip(ids, artist, song)), columns=["id", "artist", "title"]
        ).set_index("id")
        df2 = df_info.join(df_data)
        if i == 0:
            df = df2
        else:
            df = pd.concat([df, df2])
    df[p.COLUMN_LIKED] = liked
    return df


def get_analysis(
    connection: spotipy.client.Spotify, track_id: str, analysis_features: list
) -> pd.Series:
    """
    Method to retrieve audio features of a track, to be used as lambda function

    Args:
        connection: the authenticated API connection
        track_id: the ID of the track
        analysis_features: the analysis features to be retrieved

    Returns:
        _description_
    """
    track_analysis = connection.audio_analysis(track_id)["track"]
    analyses_features = pd.Series(track_analysis)[analysis_features]
    return analyses_features


def replace_playlist(connection, playlist_id: str, songs_to_add: list) -> None:
    """
    Replace all songs in a playlist with new songs.
    Note: he playlist can not be part of the profile of a user.

    Args:
        connection: the authenticated API connection
        playlist_id: The ID of the playlist to be replaced
        songs_to_add: List of song ID's to be added to the playlist
    """
    connection.playlist_replace_items(playlist_id, songs_to_add)
