from functools import wraps
import random
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from collections import namedtuple
import numpy.typing as npt
from sklearn.utils.validation import check_is_fitted
from bev_new import PositionEstimation
from ipm_evaluation.fivesafe.image_tracking.track import Track
import os
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import ConvexHull
from pathlib import Path


datapath = "../datasets/carla_dataset/"
Data = namedtuple("Data", ["image_pv", "image_tv",
                  "vehicles_pv", "vehicles_tv"])
# psi vector = richtungsvector, #hull is in [height, width]
VehicleData = namedtuple(
    "VehicleData", ['id', 'gcp', 'psi', 'img', 'bb', 'hull', 'min_area_rect'])

TrainingImage = namedtuple("TrainingImage", [
                           "image_pv_path", "image_tv_path", "vehicles_pv", "vehicles_tv"])
TrainingInstance = namedtuple("TrainingInstance", ["gcp", "psi", "hull"])


def is_vehicle_out_of_bounds(gcp, w=640, h=480):
    x, y = gcp
    if x > w or x < 0 or y > h or y < 0:
        return True
    return False


def get_vehicles_from_ts(data, ts):
    return data[ts]


def get_vehicles_from_random_ts(data):
    idx_ts = random.choice(list(data.keys()))
    return idx_ts, data[idx_ts]


def filter_vehicles_not_in_img(data, w=640, h=480):
    return [vehicle for vehicle in data if not is_vehicle_out_of_bounds(vehicle['gcp'], w, h)]


def get_random_vehicle(data):
    idx = random.randint(0, len(data)-1)
    return data[idx]


def get_vehicle_ids(data):
    return [vehicle['id'] for vehicle in data]


def filter_tv_vehicles(data, vehicle_ids):
    return [vehicle for vehicle in data if vehicle['id'] in vehicle_ids]


def cut_mask(bb, image):
    bb2d = bb_to_2d(np.array(bb))
    xmin, ymin, xmax, ymax = bb2d
    image_cropped = np.asarray(image)[ymin:ymax, xmin:xmax]
    return image_cropped


def get_mask(bb, image_cropped, image_seg):
    rgba, counts = np.unique(
        image_cropped.reshape(-1, 4), axis=0, return_counts=True)
    # Filter street
    mask = ~np.all((rgba == np.array([1, 94, 110, 255])) | 
                   (rgba == np.array([1, 65, 111, 255])) |
                   (rgba == np.array([1, 183, 117, 255]))
                   , axis=1)
    rgba = rgba[mask]
    counts = counts[mask]
    target_value = rgba[np.argmax(counts)]
    mask = np.all(np.asarray(image_seg) == target_value, axis=-1)
    return mask


def get_hull(mask):
    object_pixels = np.column_stack(np.where(mask))
    hull = cv2.convexHull(object_pixels)
    hull = hull.squeeze()
    return hull


def bb_to_2d(bb):
    x_min, x_max = np.min(bb[:, 0]), np.max(bb[:, 0])
    y_min, y_max = np.min(bb[:, 1]), np.max(bb[:, 1])
    return x_min, y_min, x_max, y_max


def plot_2d_box(bb):
    x_min, y_min, x_max, y_max = bb_to_2d(bb)
    plt.plot([x_min, x_min], [y_min, y_max], color='red')
    plt.plot([x_max, x_max], [y_min, y_max], color='red')
    plt.plot([x_min, x_max], [y_min, y_min], color='red')
    plt.plot([x_min, x_max], [y_max, y_max], color='red')


def plot_3d_box(bb):
    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
             [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]
    for edge in edges:
        plt.plot(bb[edge, 0], bb[edge, 1], color='blue')


def plot_gcp(gcp):
    _ = plt.scatter(gcp[0], gcp[1], color='cyan', marker='*', s=30)


def plot_psi(gcp, psi):
    x, y = gcp
    x2, y2 = psi
    _ = plt.plot([x, x2], [y, y2], color='red', lw=2)


def plot_hull(hull):
    plt.plot(hull[:, 1], hull[:, 0], color='orange')


def append_hull(base_url, data):
    fname = base_url+data[0]['img']
    fname_seg = base_url + 'output/seg/' + fname.split('/')[-1]
    image = Image.open(fname)
    image_seg = Image.open(fname_seg)

    # _ = plt.imshow(image)
    # _ = plt.imshow(image_seg)
    for vehicle in data:
        image_cropped = cut_mask(vehicle['bb'], image_seg)
        mask = get_mask(vehicle['bb'], image_cropped, image_seg)
        hull = get_hull(mask)
        vehicle['hull'] = hull
    return data


def append_min_area_rect(data):
    for vehicle in data:
        min_area_rect = cv2.minAreaRect(vehicle['hull'])  # Maybe need rotation
        box = cv2.boxPoints(min_area_rect)
        box = np.intp(box)
        # box = np.vstack((box, box[0, :]))
        vehicle['min_area_rect'] = box
    return data


def plot_min_area_rect(box):
    box = np.vstack((box, box[0, :]))
    plt.plot(box[:, 1], box[:, 0], color='cyan')


def get_data(ts: int, base_url: str = './', w: int = 640, h: int = 480):
    # Load Pickle Data
    with open(base_url + 'output/pv/data.pickle', 'rb') as handle:
        data_pv = pickle.load(handle)

    with open(base_url + 'output/tv/data.pickle', 'rb') as handle:
        data_tv = pickle.load(handle)

    # Perspective View
    vehicles_at_ts = get_vehicles_from_ts(data_pv, ts)
    vehicles_at_ts_filtered = filter_vehicles_not_in_img(vehicles_at_ts, w, h)
    vehicles_pv = append_hull(base_url, vehicles_at_ts_filtered)
    vehicles_pv = append_min_area_rect(vehicles_at_ts_filtered)
    fname_pv = base_url + vehicles_at_ts_filtered[0]['img']
    image_pv = Image.open(fname_pv)

    # Top View
    vehicles_at_ts_tv = get_vehicles_from_ts(data_tv, ts)
    v_ids = get_vehicle_ids(vehicles_pv)
    vehicles_tv = filter_tv_vehicles(vehicles_at_ts_tv, v_ids)
    fname_tv = base_url+vehicles_tv[0]['img']
    image_tv = Image.open(fname_tv)

    # convert to namedtuple
    # vehicles_pv = [VehicleData(**vehicle) for vehicle in vehicles_pv]
    # vehicles_tv = [VehicleData(**vehicle) for vehicle in vehicles_tv]

    return Data(image_pv, image_tv, np.array(vehicles_pv), np.array(vehicles_tv))


def get_training_data(dataset_path: Path) -> list[TrainingInstance]:
    with open(dataset_path / "output/pv/data.pickle", "rb") as f:
        data_pv = pickle.load(f)
    with open(dataset_path / "output/tv/data.pickle", "rb") as f:
        data_tv = pickle.load(f)

    assert data_pv.keys() == data_tv.keys()

    sample_image = Image.open(dataset_path / data_pv[1][0]["img"])
    image_width, image_height = sample_image.size
    all_data = []

    for key in data_pv.keys():
        pass

    return all_data


def pad_to_n(hull: npt.NDArray, n_points: int = 32) -> npt.NDArray:
    """Pad hull to always have n_points coordinates

    Args:
        hull (np.ndarray): shape: [n, 2]
        n_points (int, optional): _description_. Defaults to 32.

    Returns:
        npt.NDArray: Padded hull
    """
    # Pads the convex hull to always have 32 points
    # Can't clip points yet
    to_pad = n_points - hull.shape[0]
    assert (to_pad >= 0)  # Can't remove points yet.
    if to_pad > 0:
        # takes random points from the hull and adds a new point between point n and n+1
        point_indices = np.random.randint(0, hull.shape[0] - 2, size=to_pad)
        p1s = hull[point_indices]
        p2s = hull[point_indices+1]
        new_points = p2s + ((p1s-p2s) / 2)
        hull = np.vstack((hull, new_points))
    return hull


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
        X_padded = np.empty((len(X), 32, 2))
        for i, hull in enumerate(X):
            hull = np.array(hull)
            to_pad = self.n_points - hull.shape[0]
            assert (to_pad >= 0)  # "Can't remove points yet."
            if to_pad > 0:
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


def plot_predictions(data: Data, classifier: Pipeline, pipeline: Pipeline | None) -> Figure:
    """Plots hull, true groundpoint and predicted groundpoint

    Args:
        hull (npt.ArrayLike): variable length hull coordinates are passed through the pipeline
        true_gp (npt.ArrayLike): GP coordinates
        classifier (_type_): Classifier to call predict() on
    """
    fig, axis = plt.subplots(figsize=(17, 17))
    fig.set_frameon(False)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.imshow(data.image_pv)

    for vehicle in data.vehicles_pv:
        # copy otherwise the underlying data is mutated
        hull = vehicle["hull"].copy()
        true_gp = vehicle["gcp"]

        if pipeline:
            to_predict = pipeline.transform([hull])
        else:
            to_predict = [hull]
        predicted_gp = classifier.predict(to_predict)  # [1, 2]

        predicted_gp = predicted_gp[0]
        error = mean_squared_error(true_gp, predicted_gp)
        axis.scatter(predicted_gp[0], predicted_gp[1],
                     color="red", label="Predicted gcp")

        hull[:, [0, 1]] = hull[:, [1, 0]]  # swap coordinates
        # otherwise the connection between the first and last point is not plotted
        hull = np.vstack((hull, hull[0]))

        axis.scatter(hull[:, 0], hull[:, 1], marker='o',
                     color="blue", linewidth=0.3)
        axis.scatter(true_gp[0], true_gp[1], marker="*",
                     color="green", label="True gcp")

        line_collection = LineCollection(
            hull[np.newaxis, :, :], colors="k", linestyle="solid")
        axis.add_collection(line_collection)
        axis.set_axis_off()

    return fig


def fit_and_score(classifiers: list[tuple[Pipeline, Pipeline | None]], X, y, random_state: int) -> list[dict]:
    """Fits every model and calculates the respective test mean squared error

    Args:
        classifiers (list[tuple[Pipeline, Pipeline  |  None]]): Classifier pipeline pairs
        X (_type_): Features
        y (_type_): labels
        random_state (int): Random state for the train test split and models

    Returns:
        list[dict]: Dict{"classifier": sklearn classifier, "pipeline": pipeline,
        "train_mse": train mean squared error, "test_mse": test mean squared error}
    """
    eval_data = []

    for classifier, pipeline in tqdm(classifiers):
        if pipeline:
            X_transformed = pipeline.fit_transform(X.copy())
        else:
            X_transformed = X.copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, train_size=0.8, random_state=random_state)

        classifier.fit(X_train, y_train)

        y_pred_train = classifier.predict(X_train)
        train_mse = mean_squared_error(y_train, y_pred_train)

        y_pred_test = classifier.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred_test)

        eval_data.append({"classifier": classifier,
                          "pipeline": pipeline,
                          "train_mse": train_mse,
                          "test_mse": test_mse,
                          })

    return sorted(eval_data, key=lambda x: x["test_mse"])


def predict_and_plot(hull: npt.NDArray, true_gcp: npt.NDArray, image_size: tuple[int, int] | None,  predicted_gcp: np.ndarray | None = None):
    """Plots hull, true and predicted gcp

    Args:
        hull (npt.NDArray): Hull coordinates in shape [n, 2]
        true_gcp (npt.NDArray): shape [2,]
        predicted_gcp (_type_, optional): shape [2, ]. Defaults to None.
        scaled (bool, optional): True if hulll coordinates need to be scaled up to image dimensions to match gcp coordinates. Defaults to True.

    Returns:
        _type_: _description_
    """
    figure, axis = plt.subplots()
    hull = hull.copy()  # copy hull so the underlying data is not changed
    if image_size is not None:
        image_width, image_height = image_size
        hull *= np.array([image_height, image_width])  # scale up coordinates
    hull = ConvexHull(hull)
    points = hull.points

    axis.scatter(points[:, 0], points[:, 1], marker='o',
                 color="blue", linewidth=0.3, label="Hull coordinates")
    axis.scatter(true_gcp[1], true_gcp[0], color="green",
                 marker="*", label="True gcp")

    if predicted_gcp is not None:
        axis.scatter(predicted_gcp[1], predicted_gcp[0],
                     color="red", label="Predicted gcp")

    lines = [hull.points[simplex] for simplex in hull.simplices]
    line_collection = LineCollection(lines, colors="k", linestyle="solid")
    axis.add_collection(line_collection)
    axis.legend(loc="lower left")
    axis.set_title("Hull and ground contact point(gcp)")
    axis.spines[["right", "top"]].set_visible(False)
    # axis.set_axis_off()
    return axis


class ShiftedCenter(BaseEstimator):
    """Takes the center of the given hull and shifts it by the fitted vector.

    Args:
        BaseEstimator (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        For each hull: Assume center_point + delta = gcp => delta = gcp - center_point
        1. Calculate center point
        2. Calculate delta
        3. Take the mean of all deltas and store it for later use
        Args:
            X (np.ndarray): _description_
            y (np.ndarray): _description_
        """
        y = y.copy()
        y[:, [0, 1]] = y[:, [1, 0]]  # swap y coordinates to match hull coords
        deltas = np.empty((len(X), 2))
        for i, hull in enumerate(X):
            center = hull.mean(axis=0)
            gcp = y[i]
            delta = gcp - center
            deltas[i] = delta

        self.delta = deltas.mean(axis=0)
        self.is_fitted_ = True
        return self

    def predict(self, X: npt.NDArray) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        predictions = np.empty((len(X), 2))
        for i, hull in enumerate(X):
            center = hull.mean(axis=0)
            gcp = self.delta + center
            predictions[i] = gcp

        # swap back to expected coordinates
        predictions[:, [0, 1]] = predictions[:, [1, 0]]
        return predictions


class HardCodedEstimator(BaseEstimator):
    def __init__(self, homography_path: str, scaling_factor: int = 1) -> None:
        super().__init__()
        self.homography_path = homography_path
        self.scaling_factor = scaling_factor

    def fit(self, X, y):
        self.position_estimator_ = PositionEstimation(
            self.homography_path, self.scaling_factor)
        self.is_fitted_ = True
        label = 2  # car
        self.track_ = Track(None, label, 1, 1, 1)
        return self

    def predict(self, X: list[list[tuple[float, float]]]):
        """X is a list of hulls

        Args:
            X (list): _description_
        """
        check_is_fitted(self, "is_fitted_")
        predictions = np.empty((len(X), 2))
        for i, hull in enumerate(X):
            point, _, _ = self.position_estimator_.map_entity_and_return_relevant_points(
                self.track_, hull)
            point = self.position_estimator_.invert_homography(point)
            predictions[i] = point
        return predictions


#type ignore
def time_fn(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f"{f.__name__} took: {end-start:.2f}s")
        return result