import streamlit as st
import pandas as pd

def load_data(parquet_files):
    dataframes_list = []
    column_set = set()
    appliance_names = []

    for uploaded_file in parquet_files:
        # Extract the appliance name from the file name
        appliance_name = uploaded_file.name.split('.')[0]
        appliance_names.append(appliance_name)
    
        temp = pd.read_parquet(uploaded_file)
       # Identify new columns to add
        new_columns = [col for col in temp.columns if col not in column_set]
        if new_columns:
            column_set.update(new_columns)
            dataframes_list.append(temp[new_columns])

        # Concatenate dataframes
        df = pd.concat(dataframes_list, axis=1)
    return {"dataframe": df, "column_names": appliance_names}

def app():  
    st.title("Appliance Load Data Analysis")
    st.markdown("""
        Welcome to the Appliance Load Data Analysis app. This application allows you to upload and analyze appliance load data, 
        and navigate through different analysis tools such as Power Forecasting, and Energy Optimization.
        Follow the instructions below to get started.
    """)

    st.header("Upload Appliance Load Data")
    
    st.markdown("### Instructions:")
    st.markdown("""
        1. Ensure your files are correctly structured and named.
        2. Click the "Upload Files" button to select files from your local machine.
    """)

    # File upload
    parquet_files = st.file_uploader("Upload Parquet Files", type=["parquet"], accept_multiple_files=True, key="parquet_files")

    if st.button("Fetch Data", key="fetch_data"):
        if parquet_files:
            data = load_data(parquet_files)
            if not data["dataframe"].empty:
                st.success("Data Loaded Successfully")
                with st.expander("Preview Data"):
                    st.write(data["dataframe"].head())
                    # Store data in session state
                    st.session_state['uploaded_data'] = data["dataframe"]
                    st.session_state['appliance_names'] = data["column_names"]
            else:
                st.error("No valid data found.")
        else:
            st.error("Please upload the parquet files.")
    else:
        st.write("Select a section from the sidebar to get started.")

if __name__ == "__main__":
    app()