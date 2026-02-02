import streamlit as st
import joblib
import numpy as np
import os

# 1. Load Local Data (The "Mapping" from CSV/NPY)
# We use .astype(str) to ensure IDs are searchable as text
data_dict = np.load('data/names_onehots.npy', allow_pickle=True).item()
local_names = data_dict['names'].astype(str) 
local_features = data_dict['onehots']

# 2. Load the Brain
model = joblib.load('models/toxicity_model.pkl')

st.title("🛡️ Instant Chemical Safety Checker")
st.write("Checking household chemical safety using local AI model.")

# User Input
chemical_name = st.text_input("Enter ID (e.g., NCGC00260230-01):")

if st.button("Analyze Impact"):
    # Clean the input to remove any accidental spaces
    search_query = chemical_name.strip()
    
    if search_query in local_names:
        # Get the index (position) of the ID
        # [0][0] gets the first matching index found
        idx = np.where(local_names == search_query)[0][0]
        
        # Grab features and ensure they are 2D for the Random Forest
        features = local_features[idx].reshape(1, -1)
        
        # Predict using the saved model
        prediction = model.predict(features)
        
        if prediction == 1:
            st.error(f"⚠️ {search_query}: Predicted Impact - TOXIC")
        else:
            st.success(f"✅ {search_query}: Predicted Impact - SAFE")
    else:
        st.warning(f"ID '{search_query}' not found in the local training set.")
        st.info("Try checking the ID in your data/names_labels_test.csv file.")
