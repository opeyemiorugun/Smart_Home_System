import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from utils import minimize_standby, load_balancing, lifetime_extension, minimize_energy_cost, minimize_noise, optimize_energy_usage, plot_optimization_results

def app():
    st.title("Energy Optimization")

    original_data = st.session_state['uploaded_data']  # Assuming you have the original data stored here
    appliance_names = st.session_state['appliance_names']

    st.markdown("""
        ## Select Optimization Goals
        Choose the goals you want to achieve with the optimization. 
        You can select multiple goals for a comprehensive optimization.
    """)

    goals = st.multiselect(
        "Select Optimization Goals", [
            "Minimize Standby Power Consumption", 
            "Load Balancing", 
            "Minimize Noise Disruption", 
            "Appliance Lifetime Extension",
            "Minimize Energy Costs"
        ])

    st.markdown("""
        ## Select Appliances for Optimization
        Choose the appliances you wish to optimize. 
        You can select multiple appliances for optimization.
    """)

    selected_appliances = st.multiselect("Select appliances to optimize", appliance_names, default=appliance_names[1:])

    usage_guidelines = {
        '24_inch_lcd': {'max_daily_usage': 10, 'cool_down_period': timedelta(hours=1)},
        'dishwasher': {'max_daily_usage': 3, 'cool_down_period': timedelta(hours=2)},
        'fridge_freezer': {'max_daily_usage': 24, 'cool_down_period': timedelta(hours=0)},
        'home_theatre_amp': {'max_daily_usage': 5, 'cool_down_period': timedelta(hours=1)},
        'washer_dryer': {'max_daily_usage': 2, 'cool_down_period': timedelta(hours=2)},
    }

    if st.button("Run Optimization"):
        # Merge or align original data with forecasted data
        pre_optimization = original_data.copy()
        optimized_data, changes_logs = optimize_energy_usage(
            pre_optimization, goals, selected_appliances, 
            usage_guidelines
        )

        st.write("### Optimized Results")
        plot_optimization_results(pre_optimization, optimized_data,selected_appliances)

        st.session_state['optimized_data'] = optimized_data

        def display_changes_logs(changes_logs):
            st.write("### Optimization Changes Logs")
            for goal, log in changes_logs:
                st.subheader(f"**Goal:** {goal}")
                if log:
                    log_df = pd.DataFrame(log, columns=['Appliance', 'Time_Changed'])
                    st.write(log_df)
                else:
                    st.write("No changes were made for this goal.")
        display_changes_logs(changes_logs)

        # print("Changes Log:", changes_logs)

if __name__ == "__main__":
    app()

