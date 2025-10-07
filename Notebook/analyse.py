# === Step 1: Import libraries === 
from sentence_transformers import SentenceTransformer, util 
import numpy as np
import pandas as pd
import os
import nltk
# nltk.download('stopwords') # Decommenter si besoin
from nltk.corpus import stopwords
import re
import string
import torch 

# === Step 2: Define competency framework (blocks) ===
def load_competency_block():

    block_path = os.path.join("OneDrive", "Bureau", "ING5", "NLP", "05 - Project", "Project_NLP_MORIN_DOAT_MOUTON_LAMBERT_ROBERT_MAEDER_KFOURI", "Data", "Competency_block.csv")
    block_df = pd.read_csv(block_path)

    # Transform in dictionnary
    block_dict = block_df.set_index('job').to_dict('index')

    # Séparer les compétences en liste
    block_dict = {job: val['competency'].split('; ') for job, val in block_dict.items()}
    
    return block_dict

# === Step 3: Cleaning the user input ===
def niveau(input_niveau):
    if input_niveau == 1:
        return "mauvais"
    elif input_niveau in [2, 3]:
        return "moyen"
    elif input_niveau == 4:
        return "bon"
    elif input_niveau == 5:
        return "très bon"
    
def cleaning_user_input(user_input_df):
    # Add a new column based on the skills level of the user input
    skills_text = f"Skills level : Python niveau {niveau(user_input_df['python_level'].iloc[0])}, SQL niveau {niveau(user_input_df['sql_level'].iloc[0])}, "
    skills_text = skills_text + f"Skills level : HTML niveau {niveau(user_input_df['html_level'].iloc[0])}, CSS niveau {niveau(user_input_df['css_level'].iloc[0])}, "
    skills_text = skills_text + f"Skills level : Hadoop niveau {niveau(user_input_df['hadoop_level'].iloc[0])}, {niveau(user_input_df['cloud_level'].iloc[0])} connaissance infrastructutre cloud."

    user_input_df["skills_text"] = skills_text
    user_input_df = user_input_df.drop(columns=["python_level", "sql_level", "html_level", "css_level", "hadoop_level", "cloud_level"])

    column_cleaning = user_input_df.columns

    #List of stop words
    stop_words = set(stopwords.words('french'))

    for col in column_cleaning:
        # To lower case
        user_input_df[col] = user_input_df[col].astype(str).str.lower()
        
        #Delete punctuation
        user_input_df[col] = user_input_df[col].apply(
            lambda x: re.sub(f"[{string.punctuation}]", " ", x)
        )
        
        #Delete stopwords
        user_input_df[col] = user_input_df[col].apply(
            lambda x: " ".join([word for word in x.split() if word not in stop_words])
        )

    # Transform user input in list
    user_input = []

    for col in column_cleaning:
        user_input.append(user_input_df[col].iloc[0])
        
    return user_input
    
# === Step 4: Load SBERT model for embeddings === 
def load_model():
    model = SentenceTransformer("multi-qa-mpnet-base-dot-v1") 
    
    return model
 
# === Step 5: Encode user inputs ===
def embedding_user_input(user_input_df):
    user_input_clean = cleaning_user_input(user_input_df)
    model = load_model() 
    
    user_embeddings = model.encode(user_input_clean, convert_to_tensor=True)
    
    return user_embeddings, model

# === Step 6: Calculate semantic similarity for each block === 
def semantic_analysis(user_input_df):
    block_dict = load_competency_block()
    user_embeddings, model = embedding_user_input(user_input_df)
    block_scores = {}
    
    for block, competencies in block_dict.items(): 
        # Encode competency block phrases 
        block_embeddings = model.encode(competencies, convert_to_tensor=True) 
        
        # Compare each user input to competencies using cosine similarity 
        similarities = util.cos_sim(user_embeddings, block_embeddings) 
        
        # Take max similarity per user input and average across inputs 
        max_similarities = [float(sim.max()) for sim in similarities]   
        block_score = np.mean(max_similarities) 
        
        block_scores[block] = block_score 

    # Obtain top 3 job similarity
    top_3_blocks = sorted(block_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    return top_3_blocks

# === main ===
def main(user_input_df):
    return semantic_analysis(user_input_df)