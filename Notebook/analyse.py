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
    # block_path = os.path.join("/","Users", "antonindoat", "Desktop", "Project_NLP_MORIN_DOAT_MOUTON_LAMBERT_ROBERT_MAEDER_KFOURI", "Data", "Competency_block.csv")
    block_path = os.path.join("OneDrive", "Bureau", "ING5", "NLP", "05 - Project", "Project_NLP_MORIN_DOAT_MOUTON_LAMBERT_ROBERT_MAEDER_KFOURI", "Data", "Competency_block.csv")
    block_df = pd.read_csv(block_path)

    # Transform in dictionnary
    block_dict = block_df.set_index('Job').to_dict('index')

    # Séparer les compétences en liste
    block_dict = {job: val['Competences'].split('; ') for job, val in block_dict.items()}
    
    return block_dict

# === Step 3: Cleaning the user input ===
def cleaning_user_input(user_input_df):
    column_cleaning = user_input_df.columns

    #List of stop words
    stop_words = set(stopwords.words('english'))

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
    
    # Helper : encode proprement un texte en 2D tensor normalisé
    def encode_text(text):
        emb = model.encode(text, convert_to_tensor=True)
        emb = emb.unsqueeze(0) if emb.dim() == 1 else emb
        return emb

    # Encode parties textuelles principales
    xp_embeddings = encode_text(user_input_clean[0]).mean(dim=0)          # expérience
    interet_embeddings = encode_text(user_input_clean[1]).mean(dim=0)     # intérêts
    qual_embeddings = encode_text(user_input_clean[2]).mean(dim=0)        # qualités

    # Encode chaque skill individuel
    skills = ["python", "sql", "html", "css", "hadoop", "cloud"]
    skill_embeddings = {s: encode_text(s) for s in skills}

    # Pondération des parties principales (texte)
    xp_w, interet_w, qual_w = 0.2, 0.4, 0.05

    # Pondération des skills techniques selon le niveau utilisateur
    def skill_weight(level):
        # Niveau d'importance selon la maîtrise
        mapping = {1: 0, 2: 0, 3: 0, 4: 0.9, 5: 1.0}
        return mapping.get(level, 0.1) 


    user_embeddings = xp_w * xp_embeddings + interet_w * interet_embeddings + qual_w * qual_embeddings
    user_embeddings = user_embeddings + skill_weight(user_input_df["python_level"].iloc[0]) * skill_embeddings["python"]
    user_embeddings = user_embeddings + skill_weight(user_input_df["sql_level"].iloc[0]) * skill_embeddings["sql"]
    user_embeddings = user_embeddings + skill_weight(user_input_df["html_level"].iloc[0]) * skill_embeddings["html"]
    user_embeddings = user_embeddings + skill_weight(user_input_df["css_level"].iloc[0]) * skill_embeddings["css"]
    user_embeddings = user_embeddings + skill_weight(user_input_df["hadoop_level"].iloc[0]) * skill_embeddings["hadoop"]
    user_embeddings = user_embeddings + skill_weight(user_input_df["cloud_level"].iloc[0]) * skill_embeddings["cloud"]

    user_embeddings = torch.nn.functional.normalize(user_embeddings, p=2, dim=0)
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