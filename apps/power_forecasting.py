import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, forecast_power, calculate_metrics, plot_partial_forecast_vs_actual, plot_feature_importance

def app():
    st.title("Power Forecasting")

    if 'uploaded_data' not in st.session_state:
        st.write("Please upload data on the Home page first.")
        return

    data = st.session_state['uploaded_data']
    appliance_names = st.session_state['appliance_names']
    st.write("Data loaded successfully!")

    # Allow users to select appliances for forecasting
    selected_appliances = st.multiselect(
        'Select appliances to forecast', appliance_names, default=appliance_names[:1]
    )

    if not selected_appliances:
        st.write("Please select at least one appliance to forecast.")
        return

    model = load_model(selected_appliances)

    horizon_options = ["Next day", "Next week", "Next month"]
    horizon = st.selectbox("Select forecasting horizon", horizon_options)

    horizon_mapping = {"Next day": 1, "Next week": 7, "Next month": 30}
    forecast_horizon = horizon_mapping[horizon]

    # Identify columns to drop (unselected appliances)
    columns_to_drop = set()
    unselected_appliances = set(appliance_names) - set(selected_appliances)
    for appliance in unselected_appliances:
        columns_to_drop.update([col for col in data.columns if col.startswith(f'{appliance}_')])

    # Drop unselected appliance-specific columns
    data_filtered = data.drop(columns=columns_to_drop)

    with st.spinner("Generating forecasts..."):
        predictions = forecast_power(data_filtered, selected_appliances, model, forecast_horizon)

    actuals = data_filtered.iloc[-forecast_horizon:]
     # Ensure the indices match
    actuals = actuals.reindex(predictions.index)

    st.write("Partial Plot of Forecast vs Actual")
    plot_partial_forecast_vs_actual(actuals, predictions, selected_appliances, forecast_horizon)

    mae, rmse = calculate_metrics(actuals, predictions)
    st.write(f"MAE: {mae}")
    st.write(f"RMSE: {rmse}")

    if hasattr(model, "feature_importances_"):
        st.write("Feature Importance")
        plot_feature_importance(model.feature_importances_, data_filtered.columns)
    else:
        st.write("Model does not support feature importance.")

    st.session_state['forecasted_data'] = predictions

if __name__ == "__main__":
    app()
