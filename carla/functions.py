import carla
import math
import random
import time
import queue
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import numpy.typing as npt
from dataclasses import dataclass

def build_intrinsic_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def prepare_pickle_carla_transform(cucumber):
    x = cucumber.location.x
    y = cucumber.location.y
    z = cucumber.location.z
    pitch = cucumber.rotation.pitch
    yaw = cucumber.rotation.yaw
    roll = cucumber.rotation.roll
    return [x, y, z, pitch, yaw, roll]

def unpickle_to_carla(fresh_pickle):
    cucumber = carla.Transform()
    cucumber.location.x = fresh_pickle[0]
    cucumber.location.y = fresh_pickle[1]
    cucumber.location.z = fresh_pickle[2]
    cucumber.rotation.pitch = fresh_pickle[3]
    cucumber.rotation.yaw = fresh_pickle[4]
    cucumber.rotation.roll = fresh_pickle[5]

    return cucumber



def is_vehicle_out_of_bounds(gcp, w=1920, h=1080):
    x, y = gcp
    if x > w or x < 0 or y > h or y < 0:
        return True
    return False

def get_vehicles_from_random_ts(data):
    idx_ts = random.choice(list(data.keys()))
    return idx_ts, data[idx_ts]

def filter_vehicles_not_in_img(data, w=1920, h=1080):
    return [vehicle for vehicle in data if not is_vehicle_out_of_bounds(vehicle['gcp'], w, h)]

def get_random_vehicle(data):
    idx = random.randint(0, len(data)-1)
    return data[idx]

def get_vehicles_from_ts(data, ts):
    return data[ts]

def get_vehicle_ids(data):
    return [vehicle['id'] for vehicle in data]

def filter_tv_vehicles(data, vehicle_ids):
    return [vehicle for vehicle in data if vehicle['id'] in vehicle_ids]



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
    edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
    for edge in edges:
        plt.plot(bb[edge, 0], bb[edge, 1], color='blue')

def plot_gcp(gcp):
    _ = plt.scatter(gcp[0], gcp[1], color='cyan', marker='*', s=30)

def plot_psi(gcp, psi):
    x, y = gcp
    x2, y2 = psi
    _ = plt.plot([x, x2], [y, y2], color='red', lw=2)

def plot_hull(hull):
    hull = np.vstack((hull, hull[0, :]))
    plt.plot(hull[:, 1], hull[:, 0], color='orange')


def append_min_area_rect(data):
    for vehicle in data:
        min_area_rect = cv2.minAreaRect(vehicle['hull']) # Maybe need rotation
        box = cv2.boxPoints(min_area_rect)
        box = np.intp(box)
        vehicle['min_area_rect'] = box
    return data

def plot_min_area_rect(box):
    box = np.vstack((box, box[0, :]))
    plt.plot(box[:, 1], box[:, 0], color='cyan')

def get_data(ts, base_url='./', w=1980, h=1080):
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
    
    # Top View
    vehicles_at_ts_tv = get_vehicles_from_ts(data_tv, ts)
    v_ids = get_vehicle_ids(vehicles_pv)
    vehicles_tv = filter_tv_vehicles(vehicles_at_ts_tv, v_ids)
    fname_tv = base_url+vehicles_tv[0]['img']

def clean_segmask(image_seg):
    """ Removes all non-car pixels from the segmentation mask.
    Supposed to be called once per segmentation mask.
    Args:
    image_seg (_type_): Segmentation mask opened with opencv in rgba format
    Returns:
    _type_: Mask with only car pixels
    """
    cars_min = np.array([0, 0, 13, 0])
    cars_max = np.array([255, 255, 16, 255])
    mask = cv2.inRange(image_seg, cars_min, cars_max)
    image_seg[mask == 0] = [0, 0, 0, 0]
    return image_seg
 
 
# def cut_mask(bb, cleaned_mask):
#     """Sets all pixels outside of the bounding box to 0.
#     Args:
#         bb (_type_): bounding box of the current vehicle
#         cleaned_mask (_type_): Mask with all other colors expcept vehicles removed
#     Returns:
#         _type_: _description_
#     """
#     bb2d = bb_to_2d(np.array(bb))
#     xmin, ymin, xmax, ymax = bb2d
#     to_ignore = np.ones(cleaned_mask.shape[:2], dtype=bool)
#     to_ignore[ymin:ymax, xmin:xmax] = False
#     cleaned_mask = cleaned_mask.copy()
#     cleaned_mask[to_ignore] = 0
#     return cleaned_mask

def get_mask_for_most_common_color(image_seg):
    ids = np.sum(image_seg[:, :, 1:], axis=2)
    values, counts = np.unique(ids, return_counts=True)
    image_seg = np.where(ids == values[np.argmax(counts)], ids, 0)
    return image_seg

def append_hull(base_url, data):
    fname = base_url+data[0]['img']
    fname_seg = base_url + 'output/seg/' + fname.split('/')[-1]
    image_seg = cv2.imread(fname_seg, cv2.IMREAD_UNCHANGED)
    image_seg = clean_segmask(image_seg)
    for vehicle in data:
        cut_segmentation = cut_image(vehicle['bb'], image_seg)
        vehicle_segmentation = get_mask_for_most_common_color(cut_segmentation)
        hull = get_hull(vehicle_segmentation)


        vehicle['hull'] = hull
    return data

def cut_image(bb, image):
    bb2d = bb_to_2d(np.array(bb))
    xmin, ymin, xmax, ymax = bb2d
    image_cropped = np.asarray(image)[ymin:ymax, xmin:xmax]
    return image_cropped

def get_hull(mask):
    object_pixels = np.column_stack(np.where(mask))
    hull = cv2.convexHull(object_pixels)
    if hull is None: # skip all vehicles where the hull cant be calculated
        return None
    hull = hull.squeeze()
    return hull

def calculate_hull(seg_mask, bounding_box):
    cut_seg_mask = cut_image(bounding_box, seg_mask)
    return get_hull(cut_seg_mask)






