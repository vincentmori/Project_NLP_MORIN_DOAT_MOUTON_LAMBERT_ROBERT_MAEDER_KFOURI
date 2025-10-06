"""
Streamlit app for semantic job matching

This web app collects user inputs (experience, interests, technical skills)
through a form, stores them in a CSV file, and later performs semantic
analysis to identify which job best matches the user's profile.
"""
import streamlit as st
import pandas as pd
import os

# --- ID MANAGEMENT ---
# Penser a changer le chemin selon ou vous lancer 
id_path = os.path.join("OneDrive", "Bureau", "ING5", "NLP", "05 - Project", "Project_NLP_MORIN_DOAT_MOUTON_LAMBERT_ROBERT_MAEDER_KFOURI", "Data", "id.txt")
with open(id_path, "r") as f:
    last_id = int(f.read().strip())

new_id = last_id + 1
with open(id_path, "w") as f:
    f.write(str(new_id))

# --- CONFIGURATION ---
st.set_page_config(page_title="User Profile Form", layout="centered")
st.title("🧠 Discover your job in Data & AI!")
st.markdown("""
This form collects information about your professional experience, interests, and technical skills in order to discover which job in Data & AI suits you the most.  
Your responses will be saved into a CSV file for later analysis.
""")

# --- FORM ---
with st.form("profile_form"):
    experiences = st.text_area("Describe your experiences and related skills:", height=150)
    interests = st.text_area("Tell us about your interests in terms of skills:", height=150)
    
    col1, col2 = st.columns(2)
    with col1:
        python_level = st.slider("Python (1-5)", 1, 5, 3)
    with col2:
        sql_level = st.slider("SQL (1-5)", 1, 5, 3)

    submitted = st.form_submit_button("Save my responses")

# --- SAVE RESPONSES ---
if submitted:
    if experiences.strip() == "" and interests.strip() == "":
        st.warning("Please fill at least your experiences or interests before submitting.")
    else:
        st.success("✅ Your responses have been saved successfully!")

        user_data = pd.DataFrame({
            "experiences": [experiences],
            "interests": [interests],
            "python_level": [python_level],
            "sql_level": [sql_level]
        })

        # Penser a changer le chemin selon ou vous lancer 
        output_path = os.path.join("OneDrive", "Bureau", "ING5", "NLP", "05 - Project", "Project_NLP_MORIN_DOAT_MOUTON_LAMBERT_ROBERT_MAEDER_KFOURI", "Data", "User", f"{new_id}_profile.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        user_data.to_csv(output_path, index=False)

        st.info(f"💾 Your responses have been saved. Wait for the analysis. Your id: {new_id}")
        st.markdown("### 📝 Summary of your responses")
        st.dataframe(user_data)

st.markdown("---")
st.caption("Streamlit Prototype — User profile saved locally 📁")
