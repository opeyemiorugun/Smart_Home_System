import streamlit as st
from multiapp import MultiApp
from apps import home, power_forecasting, energy_optimization

# Set page config at the top of the main script
st.set_page_config(
    page_title="Appliance Load Data Analysis",
    page_icon="💡",
    layout="wide"
)

app = MultiApp()

# Add all application modules here
app.add_app("Home", home.app)
app.add_app("Power Forecasting", power_forecasting.app)
app.add_app("Energy Optimization", energy_optimization.app)

# Navigation Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Power Forecasting", "Energy Optimization"])

if page == "Home":
    home.app()
elif page == "Power Forecasting":
    power_forecasting.app()
elif page == "Energy Optimization":
    energy_optimization.app()
