# Building
- Uses [Hatch](https://hatch.pypa.io/)
- build with `hatch build`
- install `.whl` in dist folder

# Usage
The package only has two functions: `get_ridge_regressor_with_pipeline()` and `get_gradient_boosting_regressor_with_pipeline()` which return the regressor and their respective preprocessing pipeline.
To feed in [n, 2] data first preprocess with `pipeline.transform([x])` and then call `regressor.predict(x)`.