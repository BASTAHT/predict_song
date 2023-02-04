import os
import pickle

from dotenv import load_dotenv
from interpret.glassbox import ExplainableBoostingClassifier

import src.parameters as p
from src import preprocessing, spotify_utils

# Load secrets
load_dotenv()

# Authenticate spotify
connection = spotify_utils.auth_spotify(
    client_id=os.environ["CLIENT_ID"],
    client_secret=os.environ["CLIENT_SECRET"],
    redirect_uri=os.environ["REDIRECT_URI"],
    username=os.environ["USERNAME"],
)

# Load the liked and disliked dataset
liked_list_data = spotify_utils.get_playlist_data(
    connection=connection, playlist_id=os.environ["TRAIN_LIKE_PLAYLIST_ID"], liked=True
)

disliked_list_data = spotify_utils.get_playlist_data(
    connection=connection,
    playlist_id=os.environ["TRAIN_DISLIKE_PLAYLIST_ID"],
    liked=False,
)


# Preprocess the data
combined_data = preprocessing.combine_datasets(
    liked_playlist_data=liked_list_data,
    disliked_playlist_data=disliked_list_data,
    features=p.CLASSIFIER_FEATURES,
)

# Oversample the data
X, y = preprocessing.oversample(
    data=combined_data,
    k_neighbours=5,
    sampling_strategy=p.SAMPLING_STRATEGY,
    features=p.CLASSIFIER_FEATURES,
)

# Train the model and save it to a pickly file
ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X, y)

with open("saved_models/train_ebm_model.pickle", "wb") as f:
    pickle.dump(ebm, f)
