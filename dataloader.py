from pathlib import Path
from dataclasses import dataclass
from typing import Callable, NamedTuple, Any
import pickle
import logging
import cv2

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import distinctipy  # type: ignore

from utils import filter_vehicles_not_in_img, cut_mask, get_mask, get_hull  # type: ignore


radius = 3
line_thickness = 1


"""
Folder structure:
DatasetName:
    instances.pickle should contain a list of instances
    camera_metadata.pickle / camera_metadata.json
    images_pv
    images_tv
    images_seg not strictly necessary
"""


@dataclass(frozen=True)
class Instance:
    id: int  # vehicle id from carla
    ts: int  # timestamp should also be the image name in seg, pv and tv folders
    # pv
    gcp_pv: npt.NDArray[np.int_]  # [2,] Ground contact point in (width, height)
    psi_pv: npt.NDArray[np.int_]  # [2,] point for the direction of the vehicle
    bb_pv: npt.NDArray[np.int_]   # [8, 2] # 3d bounding box of the vehicle
    hull_pv: npt.NDArray[np.int_]  # [n, 2] # hull of the vehicle in (height, width)
    image_pv: str        # path to the image
    # tv
    gcp_tv: npt.NDArray[np.int_]  # [2,] Ground contact point
    psi_tv: npt.NDArray[np.int_]  # [2,] point for the direction of the vehicle
    bb_tv: npt.NDArray[np.int_]   # [8, 2] # 3d bounding box of the vehicle
    image_tv: str        # path to the image


@dataclass(frozen=True)
class Frame:
    ts: int
    perspective_view: npt.NDArray[np.uint8]
    top_view: npt.NDArray[np.uint8]
    vehicles: list[Instance]
    id_to_color: dict[int, tuple[int, int, int]]

    def annotate_images(self):
        image_pv = self.perspective_view.copy()
        image_tv = self.top_view.copy()
        image_height, _image_width, _ = image_pv.shape
        used_radius = radius
        used_line_thickness = line_thickness
        if image_height == 1080:
            used_radius *= 2
            used_line_thickness *= 2
        for vehicle in self.vehicles:
            hull = vehicle.hull_pv.copy()
            #hull = np.vstack([hull, hull[0]]).astype(np.int32)
            # swap hull axis
            hull[:, [0, 1]] = hull[:, [1, 0]]
            hull = hull.reshape(-1, 1, 2)
            color = self.id_to_color[vehicle.id]
            cv2.circle(image_pv, tuple(vehicle.gcp_pv), used_radius, color, -1)
            cv2.circle(image_tv, tuple(vehicle.gcp_tv), used_radius, color, -1)
            # cv2.arrowedLine(image_pv, tuple(vehicle.gcp_pv), tuple(
            #     vehicle.psi_pv), color, used_line_thickness)
            # cv2.arrowedLine(image_tv, tuple(vehicle.gcp_tv), tuple(
            #     vehicle.psi_tv), color, used_line_thickness)
            cv2.polylines(image_pv, [hull], True, color, used_line_thickness)
        return image_pv, image_tv

    @property
    def hulls(self):
        return [vehicle.hull_pv for vehicle in self.vehicles]

    def annotate_gcp_prediction(self, image: npt.NDArray[np.uint8], predictions: npt.NDArray[np.float_]):
        assert len(predictions) == len(self.vehicles)
        image = image.copy()
        predictions = predictions.astype(np.int_)
        used_radius = radius
        used_line_thickness = line_thickness
        image_height, _image_width, _ = image.shape
        if image_height == 1080:
            used_radius *= 2
            used_line_thickness *= 2
        for prediction, vehicle in zip(predictions, self.vehicles):
            color: tuple[int, int, int] = self.id_to_color[vehicle.id]
            cv2.drawMarker(image, prediction, color,
                           cv2.MARKER_STAR, used_radius*3, used_line_thickness)
        return image
    
    def annotate_psi_prediction(self, image: npt.NDArray[np.uint8], predicted_psi: npt.NDArray[np.float_], predicted_gcp: npt.NDArray[np.float_]):
        assert len(predicted_psi) == len(self.vehicles) == len(predicted_gcp)
        image = image.copy()
        predicted_psi = predicted_psi.astype(np.int_)
        predicted_gcp = predicted_gcp.astype(np.int_)
        used_radius = radius
        used_line_thickness = line_thickness
        image_height, _image_width, _ = image.shape
        if image_height == 1080:
            used_radius *= 2
            used_line_thickness *= 2
        for psi_prediction, gcp_prediction, vehicle in zip(predicted_psi, predicted_gcp, self.vehicles):
            color: tuple[int, int, int] = self.id_to_color[vehicle.id]
            cv2.arrowedLine(image, tuple(gcp_prediction), tuple(
                psi_prediction), color, used_line_thickness)
        return image


FrameSize = NamedTuple("FrameSize", [("width", int), ("height", int)])


class CarlaDataset(Dataset): # type: ignore
    name: str
    base_path: Path
    instances: list[Instance]
    camera_metadata: Any
    homography: npt.NDArray[np.float_]
    timestamps: list[int]
    frame_size: FrameSize
    transform: Callable | None 

    def __init__(self, instances: list[Instance], camera_metadata: Any, name: str, homography: npt.NDArray[np.float_], base_path: Path) -> None:
        super().__init__()
        self.name = name
        self.base_path = base_path
        self.instances = instances
        self.camera_metadata = camera_metadata
        self.homography = homography
        self.timestamps = list(
            sorted(set(instance.ts for instance in instances)))
        ids = set(instance.id for instance in instances)
        colors = distinctipy.get_colors(len(ids))  # type: ignore
        self.id_to_color = {id: distinctipy.get_rgb256(  # type: ignore
            color) for id, color in zip(ids, colors)}
        self.transform = None

    @classmethod
    def load_dataset(cls, datadir: str, force_reload: bool = False) -> "CarlaDataset":
        """Looks up the dataset format and loads it.

        Args:
            dataset_path (Path): Folder with the instances.pickle file

        Raises:
            FileNotFoundError: There is no dataset at the given path

        Returns:
            CarlaDataset: _description_
        """
        dataset_path = Path(datadir)
        if not dataset_path.is_dir():
            raise FileNotFoundError(
                f"Dataset path {dataset_path} does not exist")

        if (dataset_path / "output/instances.pickle").is_file() and not force_reload:
            return cls.load_new_format(datadir)
        else:
            return cls.load_old_format(datadir)

    @classmethod
    def load_new_format(cls, datadir: str) -> "CarlaDataset":
        dataset_path = Path(datadir)
        # get the folder name
        if not dataset_path.is_dir():
            raise FileNotFoundError(
                f"Dataset path {dataset_path} does not exist")

        with open(dataset_path / "output/instances.pickle", "rb") as f:
            instances = pickle.load(f)

        homography = cls.load_or_calculate_homography(
            dataset_path / "homography.pickle", instances)

        # TODO load camera metadata
        # with open(dataset_path / "camera_metadata.pickle", "rb") as f:
        #     self.camera_metadata = pickle.load(f)
        return cls(instances, None, dataset_path.name, homography, dataset_path)

    @classmethod
    def load_old_format(cls, datadir: str) -> "CarlaDataset":
        """Loads the dataset from the old format.

        Args:
            dataset_path (Path): _description_

        Returns:
            CarlaDataset: _description_
        """
        dataset_path = Path(datadir)
        if not dataset_path.is_dir():
            raise FileNotFoundError(
                f"Dataset path {dataset_path} does not exist")

        with open(dataset_path / "output/pv/data.pickle", "rb") as f:
            data_pv = pickle.load(f)

        with open(dataset_path / "output/tv/data.pickle", "rb") as f:
            data_tv = pickle.load(f)

        instances: list[Instance] = []

        for timestep in tqdm(data_pv.keys()):
            raw_data_pv = data_pv[timestep]
            raw_data_tv = data_tv[timestep]

            image_seg = Image.open(
                dataset_path / "output/seg/" / raw_data_pv[0]["img"].split("/")[-1])

            width, height = image_seg.size
            raw_data_pv = filter_vehicles_not_in_img(
                raw_data_pv, width, height)

            vehicle_ids = [vehicle["id"]
                           for vehicle in raw_data_pv]  # only use filtered vehicles
            raw_data_tv = [
                vehicle for vehicle in raw_data_tv if vehicle["id"] in vehicle_ids]

            for vehicle_pv, vehicle_tv in zip(raw_data_pv, raw_data_tv):
                gcp_pv = np.array(vehicle_pv["gcp"])
                psi_pv = np.array(vehicle_pv["psi"])
                bb_pv = np.array(vehicle_pv["bb"])
                try:
                    hull_pv = calculate_hull(image_seg, bb_pv)
                except ValueError:
                    logging.warning(
                        f"Could not calculate hull for vehicle with bounding box: {bb_pv}")
                    continue
                image_pv = vehicle_pv["img"]

                gcp_tv = np.array(vehicle_tv["gcp"])
                psi_tv = np.array(vehicle_tv["psi"])
                bb_tv = np.array(vehicle_tv["bb"])
                image_tv = vehicle_tv["img"]

                assert vehicle_pv["id"] == vehicle_tv["id"]
                vehicle_id = vehicle_pv["id"]
                instance = Instance(vehicle_id, timestep, gcp_pv, psi_pv,
                                    bb_pv, hull_pv, image_pv, gcp_tv, psi_tv, bb_tv, image_tv)
                instances.append(instance)

        homography_file = dataset_path / "homography.pickle"
        homography = cls.load_or_calculate_homography(
            homography_file, instances)

        with open(dataset_path / "output/instances.pickle", "wb") as f:
            pickle.dump(instances, f)

        return cls(instances, None, dataset_path.name, homography, dataset_path)

    @classmethod
    def load_or_calculate_homography(cls, homography_path: Path, instances: list[Instance]) -> npt.NDArray[np.float_]:
        """Loads the homography from the given path or calculates it from mapping gcps.

        Args:
            homography_path (str): Path to homography.pickle
        """
        if not homography_path.is_file():
            return cls.calculate_homography(instances)
        with open(homography_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def calculate_homography(instances: list[Instance]) -> npt.NDArray[np.float_]:
        gcp_pv = [instance.gcp_pv for instance in instances]
        gcp_tv = [instance.gcp_tv for instance in instances]
        gcp_pv = np.array(gcp_pv).astype(np.float32)
        gcp_tv = np.array(gcp_tv).astype(np.float32)
        # convex_hull_pv_indices = cv2.convexHull(gcp_pv, returnPoints=False)
        # convex_hull_pv_indices = np.array(convex_hull_pv_indices).squeeze()
    
        # assert convex_hull_pv_indices.ndim == 1
        # gcp_pv = gcp_pv[convex_hull_pv_indices]
        # gcp_tv = gcp_tv[convex_hull_pv_indices]

        homography, _ = cv2.findHomography(gcp_pv, gcp_tv)
        return np.array(homography)

    def __len__(self):
        return len(self.instances)

    def n_timestamps(self) -> int:
        return len(self.timestamps)

    def get_frame(self, ts: int) -> Frame:
        if ts not in self.timestamps:
            raise ValueError(f"Timestamp {ts} is not in the dataset.")
        instances = [
            instance for instance in self.instances if instance.ts == ts]
        image_pv = cv2.imread(str(self.base_path / instances[0].image_pv))
        image_tv = cv2.imread(str(self.base_path / instances[0].image_tv))
        # image_pv = cv2.cvtColor(image_pv, cv2.COLOR_BGR2RGB)
        # image_tv = cv2.cvtColor(image_tv, cv2.COLOR_BGR2RGB)

        return Frame(ts, np.asarray(image_pv), np.asarray(image_tv), instances, self.id_to_color)

    def get_instance(self, index: int) -> Instance:
        return self.instances[index]

    def __getitem__(self, index: int):
        instance = self.instances[index]
        if self.transform:
            instance = self.transform(instance)
        psi_vector = instance.psi_pv - instance.gcp_pv
        normalized_psi_vector = psi_vector / np.linalg.norm(psi_vector)

        return instance.hull_pv, instance.gcp_pv, normalized_psi_vector

    def __repr__(self) -> str:
        return f"CarlaDataset(name={self.name}, n_instances={len(self.instances)})"

    # shape (n, 2)
    def project_points_from_pv(self, points: npt.NDArray[np.uint]) -> npt.NDArray[np.float_]:
        """Projects the given points from the perspective view to the top view.

        Args:
            points (npt.NDArray[np.uint]): 

        Returns:
            npt.NDArray[np.float_]: 
        """
        return project_points(points, self.homography)

    # shape (n, 2)
    def project_points_from_tv(self, points: npt.NDArray[np.uint]) -> npt.NDArray[np.float_]:
        """Projects the given points from the top view to the perspective view.

        Args:
            points (npt.NDArray[np.uint]): 

        Returns:
            npt.NDArray[np.float_]: 
        """
        return project_points(points, np.linalg.inv(self.homography))


def calculate_hull(image_seg: Image.Image, bounding_box: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Calculates the hull of the vehicle.

    Args:
        image_seg (Image.Image): _description_
        bounding_box (npt.NDArray[np.int_]): _description_

    Returns:
        npt.NDArray[np.int_]: _description_
    """
    image_cropped: npt.NDArray[np.int_] = cut_mask(  # type: ignore
        bounding_box, image_seg)
    mask = get_mask(bounding_box, image_cropped, image_seg)
    hull: npt.NDArray[np.int_] = get_hull(mask)  # type: ignore
    return hull


def project_points(points: npt.NDArray[np.uint], homography: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """Projects the given points with the given homography matrix.

    Args:
        points (npt.NDArray): shape: [n, 2]
        homography (npt.NDArray): shape: [3, 3]

    Returns:
        npt.NDArray: shape: [n, 2]
    """
    points = np.array(points).astype(np.float32)
    points = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    points = np.dot(homography, points.T).T
    return points[:, :2] / points[:, 2:]


def load_all_datasets_in_folder(folder: Path) -> list[CarlaDataset]:
    datasets: list[CarlaDataset] = []
    for dataset in folder.iterdir():
        if dataset.is_dir():
            try:
                datasets.append(CarlaDataset.load_dataset(str(dataset)))
            except FileNotFoundError:
                logging.warning(f"Could not load dataset {dataset}")
    return datasets

def load_datasets(dataset_folder: str, dataset_names: list[str]) -> list[CarlaDataset]:
    datasets: list[CarlaDataset] = []
    for dataset_name in dataset_names:
        dataset_path = Path(dataset_folder) / dataset_name
        if dataset_path.is_dir():
            try:
                datasets.append(CarlaDataset.load_dataset(str(dataset_path)))
            except FileNotFoundError:
                logging.warning(f"Could not load dataset {dataset_path}")
    return datasets


def test_load_old_format():
    CarlaDataset.load_old_format("../datasets/carla_dataset")


def test_load_new_format():
    CarlaDataset.load_new_format("../datasets/carla_dataset")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    datasets = ["../datasets/carla_dataset", "../datasets/more_angles", "../datasets/camera1",
                "../datasets/camera2", "../datasets/camera3", "../datasets/camera4"]
    # for dataset in datasets:
    #     CarlaDataset.load_old_format(dataset)

    dataset = CarlaDataset.load_new_format("../datasets/test_instances")
    frame = dataset.get_frame(1)
    image_pv, image_tv = frame.annotate_images()
    # image = cv2.imread("../datasets/mini/"+frame.perspective_view)
    plt.imshow(image_pv)  # type: ignore
    plt.show()  # type: ignore
