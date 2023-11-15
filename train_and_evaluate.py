from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import mean_squared_error
from tqdm import tqdm, trange
import csv
import pprint
import numpy as np
import matplotlib.pyplot as plt
import joblib


from utils import ScaleToImage, PadHull, FlattenCoordinates, HardCodedEstimator, get_data, datapath

n_points_per_hull = 32
random_state = 41
image_width, image_height = 640, 480

pp = pprint.PrettyPrinter()


def fit_and_score(classifiers: list[tuple[Pipeline, Pipeline | None]], X, y) -> list[dict]:
    eval_data = []

    for classifier, pipeline in tqdm(classifiers, position=0):
        if pipeline:
            X_transformed = pipeline.fit_transform(X)
        else:
            X_transformed = X


        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, train_size=0.8, random_state=random_state, shuffle=True)

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




if __name__ == "__main__":

    scaled_pipeline = make_pipeline(PadHull(False, n_points_per_hull),
                                ScaleToImage(image_width, image_height),
                                FlattenCoordinates(n_points_per_hull),
                                StandardScaler())  # ridge needs standard scaler

    non_scaled_pipeline = make_pipeline(PadHull(False, n_points_per_hull),
                                        ScaleToImage(image_width, image_height),
                                        FlattenCoordinates(n_points_per_hull),
                                        StandardScaler())

    all_data = [get_data(i, datapath) for i in trange(1, 101)]
    X = [vehicle["hull"] for image_pv, image_tv, vehicles_pv,
        vehicles_tv in all_data for vehicle in vehicles_pv]
    y = [vehicle["gcp"] for image_pv, image_tv, vehicles_pv,
        vehicles_tv in all_data for vehicle in vehicles_pv]

    X = np.array(X, dtype=object)
    y = np.array(y)

    # commented out models don't work
    classifiers = [
        (Ridge(random_state=random_state), scaled_pipeline),
        # RegressorChain(SVR()),
        # MultiOutputRegressor(SVR()),
        (RegressorChain(GradientBoostingRegressor(n_estimators=100,
        random_state=random_state)), non_scaled_pipeline),
        (MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=100, random_state=random_state)), non_scaled_pipeline),
        # RegressorChain(LinearSVR()),
        # MultiOutputRegressor(LinearSVR()),
        (DecisionTreeRegressor(max_depth=15, random_state=random_state), non_scaled_pipeline),
        (HardCodedEstimator("./ipm_evaluation/conf/homography_matrix.json", 4), None)
    ]

    evaluation = fit_and_score(classifiers, X, y)
    with open("./evaluation_baseline.csv", "w") as f:
        w = csv.DictWriter(f, evaluation[0].keys())
        w.writeheader()
        w.writerows(evaluation)
    pp.pprint(evaluation)

    joblib.dump(classifiers, "classifiers.joblib")

    figure, axis = plt.subplots()
    names = [str(clf["classifier"]) for clf in evaluation]
    mses = [clf["test_mse"] for clf in evaluation]

    axis.bar(names, mses)
    figure.show()
    figure.savefig("evaluation.png")

