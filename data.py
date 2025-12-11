# data.py

import streamlit as st
import pandas as pd
# Add any other libraries you need (e.g., numpy, matplotlib) here:
# import numpy as np

# --- Configuration (if needed, keep it separate from the main UI code) ---
# It's best practice to use Streamlit's global functions or session_state for config
APP_CONFIG = {
    "title": "AI Data Analyzer",
    "version": "1.0",
    "include_colab_link": True  # **FIXED: Must be True (capitalized)**
}


# --- Main Application Logic ---
def main():
    # 1. SET THE PAGE CONFIGURATION (Optional, but recommended for a better look)
    st.set_page_config(
        page_title=APP_CONFIG["title"],
        layout="wide",  # Use 'wide' layout for more screen space
        initial_sidebar_state="expanded"
    )

    # 2. HEADER
    st.title(APP_CONFIG["title"])
    st.subheader("Your AI-Powered Tool for Data Exploration")

    # 3. FILE UPLOADER (Assuming this is an AI Data Analyzer)
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

    # 4. CONDITIONAL LOGIC BASED ON UPLOAD
    if uploaded_file is not None:
        # Read the file into a DataFrame
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")

            # Display the data
            st.markdown("---")
            st.header("1. Data Preview")
            st.dataframe(df.head())

            # Display basic stats
            st.header("2. Basic Statistics")
            st.write(df.describe())

            # Add your custom analysis/AI logic here...
            # st.header("3. AI Analysis")
            # ...
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.info("Please ensure your CSV file is correctly formatted.")

    else:
        # Message to show when no file is uploaded
        st.info("Please upload a CSV file to begin analysis.")
        
        # Optionally show the configuration info nicely, instead of as raw output
        st.markdown("---")
        st.text(f"App Version: {APP_CONFIG['version']}")


# --- Run the Main Function ---
# This ensures that all the code above runs correctly and is structured.
if __name__ == "__main__":
    main()
