import requests
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

def create_embeddings(text_list):
    r=requests.post("http://localhost:11434/api/embed",json={
                "model":"nomic-embed-text",
                "input":text_list
                })
   
    embedding=r.json()["embeddings"]
    return embedding

jsons=os.listdir("new_json")
my_list=[]
chunk_id=0


for json_file in jsons:
    with open (f"new_json/{json_file}") as f:
        content=json.load(f)
    print(f"creating embeddings for {json_file}")
    embeddings=create_embeddings([c["text"] for c in content["chunks"]])

    for i, chunk in enumerate(content["chunks"]):
        
        chunk["chunk_id"]=chunk_id
        chunk["embedding"]=embeddings[i]
        chunk_id+=1
        my_list.append(chunk)
        
df=pd.DataFrame.from_records(my_list)
joblib.dump(df,"embeddings.joblib")


