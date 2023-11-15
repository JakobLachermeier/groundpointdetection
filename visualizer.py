import gradio
import pathlib
from data.data import PerspectiveViewDataset, project_points
import joblib


classifiers = joblib.load("models/classifiers.joblib")
classifiers = {str(classifier["classifier"])
                   : classifier for classifier in classifiers}


data_path = pathlib.Path("../datasets")
all_datasets = [folder for folder in data_path.iterdir() if folder.is_dir()]
all_data: PerspectiveViewDataset | None = None
classifier_1 = None
classifier_2 = None

hull_color = (0, 255, 0)
true_gcp_color = (0, 0, 255)
predicted_gcp_color = (255, 0, 0)
radius = 3
line_thickness = 1


def process_frame(frame_number: int):
    if not all_data:
        raise gradio.Error("Load dataset first")

    current_frame = all_data.frames[frame_number]
    hulls = [vehicle.hull for vehicle in current_frame.vehicles_pv]
    predictions_1 = None
    predictions_2 = None
    if classifier_1 is not None:
        pipeline = classifier_1["pipeline"]
        classifier = classifier_1["classifier"]
        if pipeline is not None:
            preprocessed = classifier_1["pipeline"].transform(hulls)
        else:
            preprocessed = hulls
        predictions_1 = classifier.predict(preprocessed)
    if classifier_2 is not None:
        pipeline = classifier_2["pipeline"]
        classifier = classifier_2["classifier"]
        if pipeline is not None:
            preprocessed = classifier_2["pipeline"].transform(hulls)
        else:
            preprocessed = hulls
        predictions_2 = classifier.predict(preprocessed)
    
    projected_points_1 = None
    projected_points_2 = None
    if predictions_1 is not None:
        projected_points_1 = project_points(predictions_1, all_data.homography)
    if predictions_2 is not None:
        projected_points_2 = project_points(predictions_2, all_data.homography)

    pv_1 = current_frame.plot_pv(predicted_gcps=predictions_1)
    pv_2 = current_frame.plot_pv(predicted_gcps=predictions_2)
    tv_1 = current_frame.plot_tv(predicted_gcps=projected_points_1)
    tv_2 = current_frame.plot_tv(predicted_gcps=projected_points_2)
    return pv_1, tv_1, pv_2, tv_2


def prepare_dataset(dataset_path):
    global all_data
    print(f"Loading dataset {dataset_path}")
    try:
        all_data = PerspectiveViewDataset.load_or_create(pathlib.Path(dataset_path))
    except FileNotFoundError:
        raise gradio.Error(f"Dataset {dataset_path} not found")
    
    pv = all_data.frames[0].image_pv
    tv = all_data.frames[0].image_tv
    return pv, tv, pv, tv, gradio.Slider(0, len(all_data.frames)-1, visible=True)


with gradio.Blocks() as demo:
    with gradio.Row():
        with gradio.Column():
            model_1 = gradio.Dropdown(
                classifiers.keys(), label="Prediction Model 1")

            def change_model_1(m):
                global classifier_1
                classifier_1 = classifiers[m]
            model_1.change(change_model_1, inputs=model_1, outputs=None)

        with gradio.Column():
            model_2 = gradio.Dropdown(
                classifiers.keys(), label="Prediction Model 2")

            def change_model_2(m):
                global classifier_2
                classifier_2 = classifiers[m]
            model_2.change(change_model_2, inputs=model_2, outputs=None)

    with gradio.Row():
        with gradio.Column():
            image_tv_1 = gradio.Image(label="Top view 1")
        with gradio.Column():
            image_tv_2 = gradio.Image(label="Top view 2")

    with gradio.Row():
        with gradio.Column():
            image_pv_1 = gradio.Image(label="Perspective view 1")
        with gradio.Column():
            image_pv_2 = gradio.Image(label="Perspective view 2")
    with gradio.Row():
        slider = gradio.Slider(0, 10, visible=False,
                               step=1, label="Frame number")
        slider.release(process_frame, inputs=[slider], outputs=[
                       image_pv_1, image_tv_1, image_pv_2, image_tv_2])

    with gradio.Row():
        dataset_choice = gradio.Dropdown(choices=[str(folder) for folder in all_datasets])
        dataset_choice.change(prepare_dataset, inputs=dataset_choice, outputs=[
                              image_pv_1, image_tv_1, image_pv_2, image_tv_2, slider])

if __name__ == "__main__":
    demo.launch()
