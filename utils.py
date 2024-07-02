import pandas as pd
import joblib
import numpy as np
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgm
import seaborn as sns


def load_model(appliances):
    models = {}
    for appliance in appliances:
        filepath = f"../interactive_app/models/{appliance}_model.joblib"
        try:
            models[appliance] = joblib.load(filepath)
        except OSError as e:
            st.error(f"Error loading model for {appliance}: {e}")
    return models

def train_split_test(data, app):
    X_test, y_test = data.drop(columns=[app]), data[app]
    test = {"X_test": X_test, "y_test": y_test}
    return test

def forecast_power(data, selected_appliances, models, horizon):
    # Generate future dates for the forecast period
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(seconds=6), periods=horizon, freq='6s')
    
    forecast_df = pd.DataFrame(index=future_dates)

    # Loop through each selected appliance to generate forecasts
    for appliance in selected_appliances:
        model = models[appliance]
        
        # Create an empty DataFrame for future dates with the same columns as pre_processed_data
        future_df = pd.DataFrame(index=future_dates, columns=data.columns)
        
        # Concatenate historical data with future DataFrame
        full_df = pd.concat([data, future_df])
        
        # Handle empty rows in the concatenated DataFrame
        full_df = full_df.ffill().bfill()

        # Prepare the data for prediction
        app_specific = train_split_test(full_df, appliance)
        
        # Generate predictions
        forecast = model.predict(app_specific["X_test"])
        
        # Store the forecast values in the forecast_df
        forecast_df[appliance] = forecast[:horizon]

    return forecast_df

def calculate_metrics(actuals, predictions):
    actuals = actuals.reindex(predictions.index)  # Ensure indices match
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    return mae, rmse

def plot_partial_forecast_vs_actual(actuals, predictions, selected_appliances, forecast_horizon_days):
    # Ensure datetime index is correctly formatted
    actuals.index = pd.to_datetime(actuals.index)
    predictions.index = pd.to_datetime(predictions.index)

    # Define the time window for the partial plot
    end_date = actuals.index[-1]
    start_date = end_date - pd.Timedelta(days=forecast_horizon_days)

    # Filter the data for the partial plot
    actuals_filtered = actuals.loc[start_date:end_date]
    predictions_filtered = predictions.loc[start_date:end_date]

    for appliance in selected_appliances:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(actuals_filtered.index, actuals_filtered[appliance], label='Actual', color='blue')
        ax.plot(predictions_filtered.index, predictions_filtered[appliance], label='Forecast', color='orange', linestyle='--')
        ax.set_title(f'Partial Plot of Forecast vs Actual for {appliance}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Power (kW)')
        ax.legend()
        st.pyplot(fig)



def minimize_standby(df, column_names, standby_threshold=timedelta(hours=1)):
    """
    Minimize standby consumption by turning off appliances that are in standby mode for more than the threshold.

    Parameters:
    df (pd.DataFrame): DataFrame containing appliance states and standby information.
    column_names (list): List of appliance names.
    standby_threshold (timedelta): Time threshold for turning off standby appliances.

    Returns:
    pd.DataFrame: Updated DataFrame with minimized standby consumption.
    list: Log of changes made.
    """
    changes_log = []
    df_copy = df.copy()
    for index, row in df.iterrows():
        for col in column_names:
            # st.write(f'{col}---')
            standby_col = f'{col}_state_standby'
            if row[standby_col]:  # If the appliance is in standby mode
                # Calculate the start time of the threshold period
                start_time = row.name - standby_threshold
                # Filter for states within the threshold period
                previous_states = df_copy.loc[start_time:row.name]
                # Check if the appliance was continuously in standby mode
                if previous_states[standby_col].all():
                    # Turn off the appliance if it was in standby mode for the entire period
                    df_copy.at[index, f'{col}_state_active'] = False
                    df_copy.at[index, f'{col}_state_standby'] = False
                    df_copy.at[index, f'{col}_state_off'] = True
                    # st.write(f'{index} set to 0')
                    changes_log.append((col, row.name))
    
    return df_copy, changes_log

def load_balancing(df, column_names):
    def calculate_total_consumption(row):
        total_consumption = 0
        for col in column_names:
            active_col = f'{col}_state_active'
            if row[active_col]:
                total_consumption += row[col]
        return total_consumption

    df_copy = df.copy()
    # Calculate total consumption for each row
    df_copy['total_consumption'] = df_copy.apply(calculate_total_consumption, axis=1)
    peak_threshold = df_copy['total_consumption'].quantile(0.90)
    df_copy['is_peak'] = df_copy['total_consumption'] > peak_threshold

    # Precompute non-peak periods
    non_peak_indices = df_copy[~df_copy['is_peak']].index

    def reschedule_appliances(index, row):
        if row['is_peak']:
            for col in column_names:
                active_col = f'{col}_state_active'
                if row[active_col]:
                    next_non_peak_index = non_peak_indices[non_peak_indices > index].min()
                    if pd.notna(next_non_peak_index):
                        df_copy.at[next_non_peak_index, f'{col}_state_active'] = True
                        df_copy.at[next_non_peak_index, f'{col}_state_standby'] = False
                        df_copy.at[next_non_peak_index, f'{col}_state_off'] = False
                        df_copy.at[index, f'{col}_state_active'] = False
                        df_copy.at[index, f'{col}_state_standby'] = False
                        df_copy.at[index, f'{col}_state_off'] = True

    # Apply reschedule_appliances function to each row
    df_copy.apply(lambda row: reschedule_appliances(row.name, row), axis=1)
    return df_copy

def lifetime_extension(df, usage_guidelines):
    changes_log = []
    df_copy = df.copy()
    
    for appliance, guidelines in usage_guidelines.items():
        state_col = f'{appliance}_state_active'
        
        # Calculate daily usage
        df_copy['date'] = df_copy.index.date
        daily_usage = df_copy[df_copy[state_col]].groupby('date').size() * (3600 / 6)
        
        # Identify times when max daily usage is exceeded
        exceed_usage_dates = daily_usage[daily_usage > guidelines['max_daily_usage']].index
        exceed_usage_indices = df_copy[pd.Series(df_copy.index.date).isin(exceed_usage_dates) & df_copy[state_col]].index
        df_copy.loc[exceed_usage_indices, [f'{appliance}_state_active', f'{appliance}_state_standby', f'{appliance}_state_off']] = [False, False, True]
        changes_log.extend([(appliance, idx, 'exceeded max daily usage') for idx in exceed_usage_indices])
        
        # Identify insufficient cool down periods
        last_active_times = df_copy[df_copy[state_col]].index
        for idx in last_active_times:
            previous_active_time = last_active_times[last_active_times < idx].max()
            if previous_active_time and (idx - previous_active_time < guidelines['cool_down_period']):
                df_copy.loc[idx, [f'{appliance}_state_active', f'{appliance}_state_standby', f'{appliance}_state_off']] = [False, False, True]
                changes_log.append((appliance, idx, 'insufficient cool down period'))
    
    # Remove temporary columns
    df_copy.drop(columns=['date'], inplace=True)
    
    return df_copy, changes_log


def optimize_energy_usage(df, goals, column_names, usage_guidelines):
    changes_logs = []

    if "Minimize Standby Power Consumption" in goals:
        df, log = minimize_standby(df, column_names)
        changes_logs.append(("Minimize Standby Power Consumption", log))

    if "Load Balancing" in goals:
        df = load_balancing(df, column_names)

    if "Appliance Lifetime Extension" in goals:
        df, log = lifetime_extension(df, usage_guidelines)
        changes_logs.append(("Appliance Lifetime Extension", log))

    return df, changes_logs


def plot_optimization_results(pre_optimization, post_optimization, selected_appliances):
    st.title("Energy Optimization Results")

    total_pre_total_consumption = 0
    total_post_total_consumption = 0
    total_energy_savings = 0
    total_pre_peak_load = 0
    total_post_peak_load = 0

    appliance_data = []

    for appliance in selected_appliances:
        pre_optimization[f'{appliance}_total_consumption'] = pre_optimization[appliance] * (
            pre_optimization[f'{appliance}_state_active'] | pre_optimization[f'{appliance}_state_standby'])
        post_optimization[f'{appliance}_total_consumption'] = post_optimization[appliance] * (
            post_optimization[f'{appliance}_state_active'] | post_optimization[f'{appliance}_state_standby'])
        
        pre_optimization_filtered = pre_optimization[pre_optimization[f'{appliance}_state_off'] == False]
        post_optimization_filtered = post_optimization[post_optimization[f'{appliance}_state_off'] == False]
        
        pre_optimization_resampled = pre_optimization_filtered[[f'{appliance}_total_consumption']].resample('h').sum()
        post_optimization_resampled = post_optimization_filtered[[f'{appliance}_total_consumption']].resample('h').sum()
        
        pre_total_consumption = pre_optimization_resampled.sum().sum()
        post_total_consumption = post_optimization_resampled.sum().sum()
        
        energy_savings = pre_total_consumption - post_total_consumption
        energy_savings_percentage = (energy_savings / pre_total_consumption) * 100 if pre_total_consumption != 0 else 0
        
        pre_peak_load = pre_optimization_resampled.max().max()
        post_peak_load = post_optimization_resampled.max().max()
        peak_load_reduction = pre_peak_load - post_peak_load
        peak_load_reduction_percentage = (peak_load_reduction / pre_peak_load) * 100 if pre_peak_load != 0 else 0
        
        total_pre_total_consumption += pre_total_consumption
        total_post_total_consumption += post_total_consumption
        total_energy_savings += energy_savings
        total_pre_peak_load = max(total_pre_peak_load, pre_peak_load)
        total_post_peak_load = max(total_post_peak_load, post_peak_load)
        
        appliance_data.append({
            'appliance': appliance,
            'pre_total_consumption': pre_total_consumption,
            'post_total_consumption': post_total_consumption,
            'energy_savings': energy_savings,
            'pre_peak_load': pre_peak_load,
            'post_peak_load': post_peak_load,
            'peak_load_reduction': peak_load_reduction
        })

    appliance_df = pd.DataFrame(appliance_data)

    # New Section: Total Consumption
    st.header("Total Consumption Before and After Optimization")
    with st.container():
        st.write(f"### Total Consumption")
        st.write(f"Pre-Optimization: {total_pre_total_consumption:.2f} Wh")
        st.write(f"Post-Optimization: {total_post_total_consumption:.2f} Wh")
        st.write(f"Energy Savings: {total_energy_savings:.2f} Wh ({total_energy_savings / total_pre_total_consumption * 100:.2f}%)")

        # Bar Plot for Total Consumption
        total_consumption_df = pd.DataFrame({
            'Condition': ['Pre-Optimization', 'Post-Optimization'],
            'Total Consumption (Wh)': [total_pre_total_consumption, total_post_total_consumption]
        })

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Condition', y='Total Consumption (Wh)', data=total_consumption_df, ax=ax)
        ax.set_title('Total Consumption Before and After Optimization')
        ax.set_xlabel('Condition')
        ax.set_ylabel('Total Consumption (Wh)')
        st.pyplot(fig)

    # Section 1: Energy Savings
    st.header("Energy Savings")
    with st.container():
        # Chart for Energy Savings
        fig, ax = plt.subplots(figsize=(10, 6))
        appliance_df.plot(kind='bar', x='appliance', y=['pre_total_consumption', 'post_total_consumption'],stacked=True, ax=ax)
        ax.set_title('Energy Consumption Before and After Optimization')
        ax.set_xlabel('Appliance')
        ax.set_ylabel('Energy Consumption (Wh)')
        st.pyplot(fig)

        # Grid for individual appliance energy savings
        st.subheader("Individual Appliance Energy Savings")
        cols = st.columns(len(selected_appliances))
        for idx, appliance in enumerate(selected_appliances):
            with cols[idx]:
                st.write(f"### {appliance}")
                st.write(f"Pre-Optimization: {appliance_df.iloc[idx]['pre_total_consumption']:.2f} Wh")
                st.write(f"Post-Optimization: {appliance_df.iloc[idx]['post_total_consumption']:.2f} Wh")
                st.write(f"Energy Savings: {appliance_df.iloc[idx]['energy_savings']:.2f} Wh ({appliance_df.iloc[idx]['energy_savings'] / appliance_df.iloc[idx]['pre_total_consumption'] * 100:.2f}%)")

        # Heatmap for Energy Savings
        st.subheader("Heatmap of Energy Savings")
        heatmap_data = post_optimization[[f'{appliance}_total_consumption' for appliance in selected_appliances]].copy()
        heatmap_data['hour'] = heatmap_data.index.hour
        heatmap_data = heatmap_data.groupby('hour').sum()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data.T, cmap='YlGnBu', ax=ax)
        ax.set_title('Heatmap of Energy Savings')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Appliance')
        st.pyplot(fig)

    # Section 2: Peak Load Reduction
    st.header("Peak Load Reduction")
    with st.container():
        # Chart for Peak Load Reduction
        fig, ax = plt.subplots(figsize=(10, 6))
        appliance_df.plot(kind='bar', x='appliance', y=['pre_peak_load', 'post_peak_load'], stacked=True, ax=ax)
        ax.set_title('Peak Load Before and After Optimization')
        ax.set_xlabel('Appliance')
        ax.set_ylabel('Peak Load (kW)')
        st.pyplot(fig)

        # Grid for individual appliance peak load reduction
        st.subheader("Individual Appliance Peak Load Reduction")
        cols = st.columns(len(selected_appliances))
        for idx, appliance in enumerate(selected_appliances):
            with cols[idx]:
                st.write(f"### {appliance}")
                st.write(f"Pre-Optimization Peak Load: {appliance_df.iloc[idx]['pre_peak_load']:.2f} kW")
                st.write(f"Post-Optimization Peak Load: {appliance_df.iloc[idx]['post_peak_load']:.2f} kW")
                st.write(f"Peak Load Reduction: {appliance_df.iloc[idx]['peak_load_reduction']:.2f} kW ({appliance_df.iloc[idx]['peak_load_reduction'] / appliance_df.iloc[idx]['pre_peak_load'] * 100:.2f}%)")

        # Heatmap for Peak Load Reduction
        st.subheader("Heatmap of Peak Load Reduction")
        heatmap_data = post_optimization[[f'{appliance}_total_consumption' for appliance in selected_appliances]].copy()
        heatmap_data['hour'] = heatmap_data.index.hour
        heatmap_data = heatmap_data.groupby('hour').max()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data.T, cmap='YlGnBu', ax=ax)
        ax.set_title('Heatmap of Peak Load Reduction')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Appliance')
        st.pyplot(fig)
