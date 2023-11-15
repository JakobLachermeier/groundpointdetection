from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from PIL import Image
import numpy as np
import numpy.typing as npt
import cv2

hull_color = (0, 255, 0)
true_gcp_color = (0, 0, 255)
predicted_gcp_color = (255, 0, 0)
radius = 3
line_thickness = 1


@dataclass
class VehicleData:
    """
    Represents a single vehicle in a frame.
    """
    id: int  # id of the vehicle same for perspective and top view
    gcp: npt.NDArray  # [2,]
    psi: npt.NDArray  # [2,] point for the direction of the vehicle
    bb: npt.NDArray  # [4,] 3d bounding box of the vehicle
    # shape: [n, 2] Only used in perspective view
    hull: Optional[npt.NDArray] = None


@dataclass
class CameraImage:
    """Represents a single frame of the dataset.
    Each frame has a perspective view and a top view.
    The perspective view contains VehicleData instances for each vehicle.
    The top view contains VehicleData instances for each vehicle.

    Returns:
        _type_: _description_
    """
    image_pv: Path  # [height, width, 3]
    image_tv: Path  # [height, width, 3]

    vehicles_pv: list[VehicleData]
    vehicles_tv: list[VehicleData]

    def plot_pv(self, predicted_gcps: npt.NDArray | None = None) -> npt.NDArray:
        """Plots the perspective view with the hulls and gcps of the vehicles.

        Args:
            predicted_gcps (npt.NDArray | None, optional): shape: [n, 2]. The number of predictions has to match the number of instances.
            Defaults to None.

        Returns:
            npt.NDArray: _description_
        """
        image_pv = self.image_pv.copy()
        if predicted_gcps is not None:
            assert len(predicted_gcps) == len(self.vehicles_pv)
        for i, instance_pv in enumerate(self.vehicles_pv):
            # swap x and y coordinate of hull
            assert instance_pv.hull is not None
            hull = instance_pv.hull.copy()
            hull[:, [0, 1]] = hull[:, [1, 0]]
            cv2.drawContours(image_pv, [hull], 0, hull_color, line_thickness)
            center = tuple(instance_pv.gcp.astype(int))
            cv2.circle(image_pv, center, radius, true_gcp_color, -1)
            if predicted_gcps is not None:
                predicted_center = tuple(predicted_gcps[i].astype(int))
                cv2.circle(image_pv, predicted_center,
                           radius, predicted_gcp_color, -1)

        return image_pv

    def plot_tv(self, predicted_gcps: npt.NDArray | None = None) -> npt.NDArray:
        image_tv = self.image_tv.copy()
        assert len(self.vehicles_tv) == len(self.vehicles_pv)
        for i, instance_tv in enumerate(self.vehicles_tv):
            center = tuple(instance_tv.gcp.astype(int))
            cv2.circle(image_tv, center, radius, true_gcp_color, -1)
            if predicted_gcps is not None:
                predicted_center = tuple(predicted_gcps[i].astype(int))
                cv2.circle(image_tv, predicted_center,
                           radius, predicted_gcp_color, -1)
        return image_tv



class PerspectiveView(Dataset):
    def __init__(self, dataset_path: Path, transforms, target_transforms) -> None:
        super().__init__()
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        
        with open(dataset_path / "output/pv/data.pickle", "rb") as f:
            data_pv = pickle.load(f)

        with open(dataset_path / "output/tv/data.pickle", "rb") as f:
            data_tv = pickle.load(f)

        for key in data_pv.keys():
            vehicles_pv = []
            vehicles_tv = []
            raw_data_pv = data_pv[key]  # keys: id, img, gcp, psi, bb
            raw_data_tv = data_tv[key]  # keys: id, img, gcp, psi, bb

            image_pv = cv2.imread(str(dataset_path / raw_data_pv[0]["img"]))
            image_tv = cv2.imread(str(dataset_path / raw_data_tv[0]["img"]))
            image_pv = cv2.cvtColor(image_pv, cv2.COLOR_BGR2RGB)
            image_tv = cv2.cvtColor(image_tv, cv2.COLOR_BGR2RGB)

            image_seg = Image.open(
                dataset_path / "output/seg/" / raw_data_pv[0]["img"].split("/")[-1])
            assert image_pv.shape == image_tv.shape
            height, width, _ = image_pv.shape
            raw_data_pv = filter_vehicles_not_in_img(
                raw_data_pv, width, height)
            append_hull(image_seg, raw_data_pv)

            vehicle_ids = [vehicle["id"]
                           for vehicle in raw_data_pv]  # only use filtered vehicles
            raw_data_tv = [
                vehicle for vehicle in raw_data_tv if vehicle["id"] in vehicle_ids]

            # convert to numpy and remove img from raw data

            # convert to numpy and remove img from raw data
            for vehicle_pv, vehicle_tv in zip(raw_data_pv, raw_data_tv):
                vehicle_pv["gcp"] = np.array(vehicle_pv["gcp"])
                vehicle_pv["psi"] = np.array(vehicle_pv["psi"])
                vehicle_pv.pop("img")
                vehicle_tv["gcp"] = np.array(vehicle_tv["gcp"])
                vehicle_tv["psi"] = np.array(vehicle_tv["psi"])
                vehicle_tv.pop("img")
                vehicles_pv.append(VehicleData(**vehicle_pv))
                vehicles_tv.append(VehicleData(**vehicle_tv))
            dataset.append(CameraImage(
                image_pv, image_tv, vehicles_pv, vehicles_tv))

        self.frames = dataset
        self.homography = calculate_homography(dataset)

        