from functools import partial
from attr import dataclass
import gradio  # type: ignore
import pathlib
import uuid
from pathlib import Path

import numpy as np
import numpy.typing as npt
from dataloader import CarlaDataset, project_points, load_datasets, load_all_datasets_in_folder
from dataloader import Instance  # type: ignore needed to unpickle
import joblib  # type: ignore
import cv2
from typing import Literal

from utils import HardCodedEstimator


gcp_regressors = joblib.load("regressors.joblib")  # type: ignore
gcp_regressors = {str(classifier["classifier"])
                  : classifier for classifier in gcp_regressors}

psi_regressors = joblib.load("psi_regressors.joblib")  # type: ignore
psi_regressors = {str(classifier["classifier"])
                    : classifier for classifier in psi_regressors}

data_path = "../datasets"
all_datasets = {
    dataset.name: dataset for dataset in load_datasets(data_path, ["train2", "train3", "train4", "validation"])}
# all_datasets = {
#    dataset.name: dataset for dataset in load_all_datasets_in_folder(Path(data_path) / "bulk_test_sets")}
# all_datasets = {
#     dataset.name: dataset for dataset in load_datasets("./", ["carla_datagen"])}


def save_img_array_to_cache(
    arr: npt.NDArray[np.uint8], cache_dir: str, format: Literal["png", "jpg"] = "png"
) -> str:
    temp_dir = pathlib.Path(cache_dir) / str(uuid.uuid4())
    temp_dir.mkdir(exist_ok=True, parents=True)
    filename = (temp_dir / f"image.{format}").resolve()
    cv2.imwrite(str(filename), arr)  # type: ignore
    return str(filename)


gradio.processing_utils.save_img_array_to_cache = save_img_array_to_cache  # type: ignore


@dataclass
class SessionState:
    current_dataset: CarlaDataset | None = None
    classifier_1: dict | None = None  # type: ignore
    classifier_2: dict | None = None  # type: ignore
    psi_regressor_1: dict | None = None  # type: ignore
    psi_regressor_2: dict | None = None  # type: ignore


def process_frame(frame_number: int, session_state: SessionState):
    classifier_1 = session_state.classifier_1  # type: ignore
    classifier_2 = session_state.classifier_2  # type: ignore
    psi_regressor_1 = session_state.psi_regressor_1  # type: ignore
    psi_regressor_2 = session_state.psi_regressor_2  # type: ignore
    current_dataset = session_state.current_dataset
    if not current_dataset:
        raise gradio.Error("Load dataset first")
    current_frame = current_dataset.get_frame(frame_number)
    hulls = current_frame.hulls.copy()
    predictions_1 = None
    predictions_2 = None
    if classifier_1 is not None:
        pipeline = classifier_1["pipeline"]  # type: ignore
        classifier = classifier_1["classifier"]  # type: ignore
        if pipeline is not None:
            preprocessed = classifier_1["pipeline"].transform(  # type: ignore
                hulls)  # type: ignore
        else:
            preprocessed = hulls
        predictions_1 = classifier.predict(preprocessed)  # type: ignore

    if classifier_2 is not None:
        pipeline = classifier_2["pipeline"]  # type: ignore
        classifier = classifier_2["classifier"]  # type: ignore
        if pipeline is not None:
            preprocessed = classifier_2["pipeline"].transform(  # type: ignore
                hulls)
        else:
            preprocessed = hulls
        predictions_2 = classifier.predict(preprocessed)  # type: ignore

    predicted_psi_1 = None
    predicted_psi_2 = None
    if psi_regressor_1 is not None:
        pipeline = psi_regressor_1["pipeline"]
        regressor = psi_regressor_1["classifier"]
        if pipeline is not None:
            preprocessed = psi_regressor_1["pipeline"].transform(
                hulls)
        else:
            preprocessed = hulls
        predicted_psi_1 = regressor.predict(preprocessed)
    
    if psi_regressor_2 is not None:
        pipeline = psi_regressor_2["pipeline"]
        regressor = psi_regressor_2["classifier"]
        if pipeline is not None:
            preprocessed = psi_regressor_2["pipeline"].transform(
                hulls)
        else:
            preprocessed = hulls
        predicted_psi_2 = regressor.predict(preprocessed)

    pv, tv = current_frame.annotate_images()
    projected_points_1 = None
    projected_points_2 = None
    if predictions_1 is not None:
        projected_points_1 = project_points(
            predictions_1, current_dataset.homography)  # type: ignore
        pv_1 = current_frame.annotate_gcp_prediction(
            pv, predictions_1)  # type: ignore
        tv_1 = current_frame.annotate_gcp_prediction(tv, projected_points_1)
    else:
        pv_1 = pv
        tv_1 = tv
    if predictions_2 is not None:
        projected_points_2 = project_points(
            predictions_2, current_dataset.homography)  # type: ignore
        pv_2 = current_frame.annotate_gcp_prediction(
            pv, predictions_2)  # type: ignore
        tv_2 = current_frame.annotate_gcp_prediction(tv, projected_points_2)
    else:
        pv_2 = pv
        tv_2 = tv


    if predictions_1 is not None and predicted_psi_1 is not None:
        arrow_head = predictions_1 + predicted_psi_1 * 100
        pv_1 = current_frame.annotate_psi_prediction(pv_1, arrow_head, predictions_1)

    if predictions_2 is not None and predicted_psi_2 is not None:
        arrow_head = predictions_2 + predicted_psi_2 * 100
        pv_2 = current_frame.annotate_psi_prediction(pv_2, arrow_head, predictions_2)

    return pv_1, tv_1, pv_2, tv_2


def set_dataset(dataset_name: str, session_state: SessionState) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], gradio.Slider, SessionState]:
    print(f"Loading dataset {dataset_name}")
    try:
        dataset = all_datasets[dataset_name]
    except KeyError:
        raise gradio.Error(f"Dataset {dataset_name} not found")

    first_frame = dataset.get_frame(1)
    imave_pv, image_tv = first_frame.annotate_images()
    min_ts = dataset.timestamps[0]
    max_ts = dataset.timestamps[-1]
    session_state.current_dataset = dataset

    # update homography for hardcoded estimators
    for regressor in gcp_regressors.values():
        if isinstance(regressor["classifier"], HardCodedEstimator):
            regressor["classifier"].set_homography(dataset.homography)

    return imave_pv, image_tv, imave_pv, image_tv, gradio.Slider(min_ts, max_ts, interactive=True, value=0), session_state


with gradio.Blocks() as demo:
    session_state = gradio.State(SessionState())
    Image = partial(gradio.Image, type="numpy", render=False)
    image_tv_1 = Image(label="Top view 1")
    image_pv_1 = Image(label="Perspective view 1")
    image_tv_2 = Image(label="Top view 2")
    image_pv_2 = Image(label="Perspective view 2")
    slider = gradio.Slider(0, 10, interactive=False,
                           step=1, label="Frame number", render=False)

    with gradio.Group():
        dataset_choices = gradio.Radio(
            list(all_datasets.keys()), label="Dataset")

        @dataset_choices.change(inputs=[dataset_choices, session_state], outputs=[  # type: ignore
            image_pv_1, image_tv_1, image_pv_2, image_tv_2, slider, session_state])
        def change_dataset(dataset_path: str, session_state: SessionState):
            return set_dataset(dataset_path, session_state)

        with gradio.Row():
            model_1 = gradio.Radio(
                list(gcp_regressors.keys()), label="Ground point regressor 1")

            # type: ignore
            @model_1.change(inputs=[model_1, session_state], outputs=session_state)
            def change_model_1(m: str, session_state: SessionState) -> SessionState:
                new_regressor = gcp_regressors[m]
                session_state.classifier_1 = new_regressor
                return session_state

            model_2 = gradio.Radio(
                list(gcp_regressors.keys()), label="Ground point regressor 2")

            # type: ignore
            @model_2.change(inputs=[model_2, session_state], outputs=session_state)
            def change_model_2(m: str, session_state: SessionState) -> SessionState:
                new_regressor = gcp_regressors[m]
                session_state.classifier_2 = new_regressor
                return session_state

        with gradio.Row():
            psi_regressor_1 = gradio.Radio(
                list(psi_regressors.keys()), label="Psi regressor 1")
            
            @psi_regressor_1.change(inputs=[psi_regressor_1, session_state], outputs=session_state)
            def change_psi_regressor_1(m: str, session_state: SessionState) -> SessionState:
                new_regressor = psi_regressors[m]
                session_state.psi_regressor_1 = new_regressor
                return session_state
            

            psi_regressor_2 = gradio.Radio(
                list(psi_regressors.keys()), label="Psi regressor 2")
            @psi_regressor_2.change(inputs=[psi_regressor_2, session_state], outputs=session_state)
            def change_psi_regressor_2(m: str, session_state: SessionState) -> SessionState:
                new_regressor = psi_regressors[m]
                session_state.psi_regressor_2 = new_regressor
                return session_state

        with gradio.Row():
            slider.render()
            slider.release(process_frame, inputs=[slider, session_state], outputs=[  # type: ignore
                image_pv_1, image_tv_1, image_pv_2, image_tv_2])

        with gradio.Column():
            with gradio.Row():
                image_tv_1.render()
                image_tv_2.render()

            with gradio.Row():
                image_pv_1.render()
                image_pv_2.render()


if __name__ == "__main__":
    # demo.queue()
    demo.launch(server_name="0.0.0.0")  # type: ignore
