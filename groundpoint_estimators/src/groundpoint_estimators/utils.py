from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline  
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from importlib_resources import files, as_file

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
N_POINTS_PER_HULL = 100


class PadHull(BaseEstimator, TransformerMixin):
    def __init__(self, shuffle: bool, n_points: int = 32) -> None:
        """Pads each hull to have at least n_points hull coordinates.

        Args:
            shuffle (bool): If True the hull coordinates are shuffled instead of sorted
            n_points (int, optional): Number of points for each hull. Defaults to 32.
        """
        super().__init__()
        self.n_points = n_points
        self.shuffle = shuffle

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_padded = np.empty((len(X), self.n_points, 2))
        for i, hull in enumerate(X):
            hull = np.array(hull)
            to_pad = self.n_points - hull.shape[0]
            if to_pad < 0:
                hull = hull[:self.n_points]
            elif to_pad > 0:
                point_indices = np.random.randint(
                    0, hull.shape[0] - 2, size=to_pad)
                p1s = hull[point_indices]
                p2s = hull[point_indices+1]
                new_points = p2s + ((p1s-p2s) / 2)
                hull = np.vstack((hull, new_points))
            if self.shuffle:
                np.random.shuffle(hull)  # shuffle points
            X_padded[i] = hull
        return X_padded


class ScaleToImage(BaseEstimator, TransformerMixin):
    def __init__(self, width: int, height: int) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def fit(self, X):
        return self

    def transform(self, X):
        image_dimensions = np.array([self.height, self.width]).reshape(1, 2)
        return X / image_dimensions


class FlattenCoordinates(BaseEstimator, TransformerMixin):
    def __init__(self, n_coordinates: int) -> None:
        super().__init__()
        self.n_coordinates = n_coordinates

    def fit(self, X):
        return self

    def transform(self, X):
        # flatten all coordinates so they can fit into the models
        return X.reshape(-1, self.n_coordinates*2)




def get_ridge_regressor_with_pipeline():
    source_file = files("groundpoint_estimators").joinpath("ridge.joblib")
    standard_scaler_file = files("groundpoint_estimators").joinpath("standard_scaler.joblib")

    with as_file(source_file) as ridge_file, as_file(standard_scaler_file) as standard_scaler_file:
        ridge: Ridge = joblib.load(ridge_file)
        standard_scaler = joblib.load(standard_scaler_file)
        pipeline = make_pipeline(PadHull(False, N_POINTS_PER_HULL),
                                ScaleToImage(IMAGE_WIDTH, IMAGE_HEIGHT),
                                FlattenCoordinates(N_POINTS_PER_HULL),
                                standard_scaler) # ridge needs standard scaler
        return ridge, pipeline

def get_gradient_boosting_regressor_with_pipeline():
    source_file = files("groundpoint_estimators").joinpath("gradient_boosting.joblib")
    with as_file(source_file) as file_name:
        gb_regressor: MultiOutputRegressor = joblib.load(file_name)
        pipeline = make_pipeline(PadHull(False, N_POINTS_PER_HULL),
                                ScaleToImage(IMAGE_WIDTH, IMAGE_HEIGHT),
                                FlattenCoordinates(N_POINTS_PER_HULL),
                                )
        return gb_regressor, pipeline
