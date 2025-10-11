"""
Streamlit app for semantic job matching

This web app collects user inputs (experience, interests, technical skills)
through a form, stores them in a CSV file, and later performs semantic
analysis to identify which job best matches the user's profile.
"""
import streamlit as st
import pandas as pd
import os

import sys

# === Add Notebook file path to Python path ===
# Change with your path
path_to_notebook = "Notebook"

# Now you can import your analysis module
import analyse

# --- ID MANAGEMENT ---
# Change with your path
id_path = os.path.join("Data", "id.txt") 

with open(id_path, "r") as f:
    last_id = int(f.read().strip())

new_id = last_id + 1

# --- CONFIGURATION ---
st.set_page_config(page_title="User Profile Form", layout="centered")
st.title("🧠 Discover your job in IT!")
st.markdown("""
This form collects information about your professional experience, interests, qualities and technical skills in order to discover which job in IT suits you the most.  
Your responses will be saved into a CSV file for later analysis.
""")

# --- FORM ---
with st.form("profile_form"):
    experiences = st.text_area("Describe your experiences and related skills:", height=150)
    interests = st.text_area("Tell us about your interests in terms:", height=150)
    qualities = st.text_area("What are your main qualities:", height=150)
    
    col1, col2 = st.columns(2)
    with col1:
        python_level = st.slider("Python (1-5)", 1, 5, 3)
    with col2:
        sql_level = st.slider("SQL (1-5)", 1, 5, 3)
        
    col1, col2 = st.columns(2)
    with col1:
        html_level = st.slider("HTML (1-5)", 1, 5, 3)
    with col2:
        css_level = st.slider("CSS (1-5)", 1, 5, 3)
        
    col1, col2 = st.columns(2)
    with col1:
        hadoop_level = st.slider("Hadoop (1-5)", 1, 5, 3)
    with col2:
        cloud_level = st.slider("Cloud infrastructure (1-5)", 1, 5, 3)    

    submitted = st.form_submit_button("Save my responses")

# --- Print the top3 corresponding jobs ---
if submitted:
    if experiences.strip() == "" or interests.strip() == "" or qualities.strip() == "":
        st.warning("Please fill every subjecty. It is really important for a better response.")
    else:
        st.success("✅ Your responses have been saved successfully!")

        user_data = pd.DataFrame({
            "experiences": [experiences],
            "interests": [interests],
            "qualities": [qualities],
            "python_level": [python_level],
            "sql_level": [sql_level],
            "html_level": [html_level],
            "css_level": [css_level],
            "hadoop_level": [hadoop_level],
            "cloud_level": [cloud_level]
        })

        # Save the profil 
        
        # Change with your path
        output_path = os.path.join("Data", "User_input", f"{new_id}_profile.csv")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        user_data.to_csv(output_path, index=False)
        
        with open(id_path, "w") as f:
            f.write(str(new_id))
            
        # --- Spinner while analysis runs ---
        with st.spinner("🧠 Analysing your profile... Please wait a moment."):
            top_3 = analyse.main(user_data)
        
        st.success("🎯 Analysis complete!")
        
        # --- Sort in descending order --- 
        top_3_sorted = sorted(top_3, key=lambda x: x[1], reverse=True)
        
        # --- Display podium with links ---
        st.title("🏆 Your most suiting jobs in IT")

        for i, (job, score) in enumerate(top_3_sorted, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            st.markdown(f"### {emoji} *{job}* {score} **")

            # --- Automatically generate a search Welcome to the Jungle ---
            query = job.replace(" ", "%20")
            url = f"https://www.welcometothejungle.com/fr/jobs?query={query}"

            st.markdown(
                f"""
                <div style="background-color:#f9f9f9;border:1px solid #e0e0e0;border-radius:8px;padding:10px;margin-bottom:15px;">
                    <p style="margin:5px 0;">🔗 <a href="{url}" target="_blank">Voir les offres "{job}" sur Welcome to the Jungle</a></p>
                </div>
                """,
                unsafe_allow_html=True
            )


        top3_df = pd.DataFrame(top_3_sorted, columns=["Job", "Similarity"])

        # Change with your path 
        output_path = os.path.join("Data", "Response", f"{new_id}_top.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        top3_df.to_csv(output_path, index=False)