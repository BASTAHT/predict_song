import pandas as pd
from imblearn.over_sampling import SMOTE

from . import parameters as p


def combine_datasets(
    liked_playlist_data: pd.DataFrame,
    disliked_playlist_data: pd.DataFrame,
    features: list = p.CLASSIFIER_FEATURES,
):
    combined_set = pd.concat(
        (
            liked_playlist_data[features + [p.COLUMN_LIKED]],
            disliked_playlist_data[features + [p.COLUMN_LIKED]],
        )
    )
    return combined_set


def oversample(
    data: pd.DataFrame,
    k_neighbours: int = 5,
    sampling_strategy: float = 1.0,
    features: list = p.CLASSIFIER_FEATURES,
):
    """
    Method that oversamples the train set using SMOTE (Synthetic Minority Over-sampling Technique)
    The minority class if oversampled to achieve specified ratio of majority and minotory class
    :param k_neighbours: the number of nearest neighbours used in the creation of synthetic datapoints
    :param sampling_strategy: The ratio between number of minority samples (after oversampling) and majority class.
                                Default 1.0 leads to the same number of minority class as majority class samples.
    """
    oversample = SMOTE(k_neighbors=k_neighbours, sampling_strategy=sampling_strategy)
    x_smote, y_smote = oversample.fit_resample(data[features], data[p.COLUMN_LIKED])

    return x_smote, y_smote
