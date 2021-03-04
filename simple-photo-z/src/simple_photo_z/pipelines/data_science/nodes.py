import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import tqdm
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestRegressor


def process_predictions(y_pred: np.array):
    """Calculate point-estimation from probabilistic prediction in form of
    samples
    :param y_pred: 2-d np.ndarray of shape (number of test samples, ...)
    :return: np.ndarray of shape (number of test samples)
    """
    grid = np.linspace(0, 7, 701)
    point_predictions = []
    for pred in tqdm.tqdm(y_pred):
        kde = gaussian_kde(pred, 0.1)
        pdf = kde.pdf(grid)
        pp = grid[np.argmax(pdf)]
        point_predictions.append(pp)

    return np.array(point_predictions)


def get_model_features(df: pd.DataFrame) -> List:
    """Get X-train and y-train from dataframe
    :param df: pd.DataFrame
    :return: [X_train: np.ndarray, y_train: np.ndarray]
    """
    feats = [
        'ls_asinhmag_g', 'ls_asinhmag_r', 'ls_asinhmag_z',
        'ls_asinhmag_g-r', 'ls_asinhmag_g-z', 'ls_asinhmag_r-z',

        'wise_asinhmag_w1', 'wise_asinhmag_w1', 'wise_asinhmag_w1-w2',

        'ls_asinhmag_g-w1', 'ls_asinhmag_r-w1', 'ls_asinhmag_z-w1',
        'ls_asinhmag_g-w2', 'ls_asinhmag_r-w2', 'ls_asinhmag_z-w2',
    ]

    X = df[feats]
    y = df['zspec']
    X = X.replace([-np.inf, np.inf], np.nan)
    mask = X.notna().all(axis=1) & y.notna()
    return [X.loc[mask].values, y.loc[mask].values]


def train_model(
        X_train: np.ndarray, y_train: np.ndarray
) -> RandomForestRegressor:
    """Train model"""
    rf_kws = dict(
        n_estimators=288,
        min_samples_leaf=1,
        max_features='auto',
        max_samples=None,
        bootstrap=True,
        n_jobs=16,
    )
    regr = RandomForestRegressor(**rf_kws)
    regr.fit(X_train, y_train)
    return regr


def nmad_z(z_pred: np.ndarray, z_true: np.ndarray) -> float:
    """
    Calculates NMAD error using dz_norm = (z_pred - z_true)/(1 + z_true) instead of dz = (z_pred - z_true)
    :param z_pred: numpy.ndarray, shape = (number of objects), values of predicted photoz
    :param z_true: numpy.ndarray, shape = (number of objects), values of specz
    :return: float, NMAD
    """
    dz_norm = (z_pred - z_true) / (1 + z_true)
    return 1.4826 * np.median(np.abs(dz_norm))


def catastrophic_outliers_z(z_pred: np.ndarray, z_true: np.ndarray) -> float:
    """
    Calculates fraction of catastrophic outliers using dz_norm = (z_pred - z_true)/(1 + z_true) instead of
        dz = (z_pred - z_true). Catastrophic outlier is a prediction that has |dz_norm| >= 0.15
    :param z_pred: numpy.ndarray, shape = (number of objects), values of predicted photoz
    :param z_true: numpy.ndarray, shape = (number of objects), values of specz
    :return: float, fraction of catastrophic outliers
    """
    dz_norm = (z_pred - z_true) / (1 + z_true)
    n = z_true.shape[0]
    n_lower_015 = np.sum(np.abs(dz_norm) < 0.15)
    return 1 - n_lower_015 / n


def evaluate_model(
        regr: RandomForestRegressor, X_test: np.ndarray, y_test: np.ndarray
) -> None:
    """
    Prints NMAD and n>0.15
    :param regr: RandomForestRegressior
    :param X_test:
    :param y_test:
    :return:
    """
    y_pred = np.array([tree.predict(X_test) for tree in tqdm.tqdm(regr.estimators_)]).T
    y_pred = process_predictions(y_pred)
    scores = {
        "NMAD": nmad_z(y_pred, y_test),
        "n>0.15": catastrophic_outliers_z(y_pred, y_test)
    }
    logger = logging.getLogger(__name__)
    logger.info(f"Scores of a model: {scores}")
