from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import logging

def analyze_feature_importance(self, symbol):
    """Estimate feature importance using weights"""
    X, y_direction, y_return = self.prepare_data(symbol)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Recreate and load trained models (assuming they exist)
    direction_model = load_model(f"ml/ml_models/{symbol}_direction_model.keras")
    return_model = load_model(f"ml/ml_models/{symbol}_return_model.keras")

    # Extract first layer weights for feature importance estimation
    dir_weights = np.abs(direction_model.layers[0].get_weights()[0]).mean(axis=1)
    ret_weights = np.abs(return_model.layers[0].get_weights()[0]).mean(axis=1)

    # Create feature importance DataFrames
    dir_importance = pd.DataFrame(
        {"feature": X.columns, "direction_importance": dir_weights}
    ).sort_values("direction_importance", ascending=False)

    ret_importance = pd.DataFrame(
        {"feature": X.columns, "return_importance": ret_weights}
    ).sort_values("return_importance", ascending=False)

    logging.info("Direction Model Feature Importance:")
    logging.info(dir_importance)
    logging.info("\nReturn Model Feature Importance:")
    logging.info(ret_importance)

    return dir_importance, ret_importance
