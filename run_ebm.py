import os
import pickle

from dotenv import load_dotenv

import src.utils.parameters as p
from src.utils import spotify_utils

# Load secrets
load_dotenv()

# Authenticate spotify
connection = spotify_utils.auth_spotify(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    redirect_uri=os.getenv("REDIRECT_URI"),
    username=os.getenv("USERNAME"),
)

# Load the playlist to classify
classify_list_data = spotify_utils.get_playlist_data(
    connection=connection, playlist_id=os.getenv("CLASSIFY_PLAYLIST_ID"), liked=False
)

# Load and apply the model
with open("saved_models/train_ebm_model.pickle", "rb") as f:
    trained_ebm_model = pickle.load(f)

prediction = trained_ebm_model.predict(classify_list_data[p.CLASSIFIER_FEATURES])

classify_list_data["predicted"] = prediction

# Split the input songs based on their prediction
filter_liked = classify_list_data["predicted"]
predict_liked = classify_list_data[filter_liked][["artist", "title", "predicted"]]
predict_disliked = classify_list_data[~filter_liked][["artist", "title", "predicted"]]

# Write the songs predicted as liked to output playlist
spotify_utils.replace_playlist(
    connection, os.getenv("PREDICTED_LIKE_ID"), list(predict_liked.index)
)

# write the songs predicted as liked to output playlist
spotify_utils.replace_playlist(
    connection, os.getenv("PREDICTED_DISLIKE_ID"), list(predict_disliked.index)
)

# Print the aggregate results
print(
    f"{len(predict_liked)} songs were predicted as liked and {len(predict_disliked)} as disliked. "
    f"The songs were added to automatically generated playlists."
)
