import requests
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from config import api_key

def create_embeddings(text_list):
    r=requests.post("http://localhost:11434/api/embed",json={
                "model":"nomic-embed-text",
                "input":text_list
                })
   
    embedding=r.json()["embeddings"]
    return embedding

# def inference(prompt):
#     r=requests.post("http://localhost:11434/api/generate",json={
#                 "model":"llama3.2",
#                 "prompt":prompt,
#                 "stream": False,

#                 })
#     response=r.json()
#     print(response)
#     return response



client = genai.Client(api_key=api_key)
def inference_genai(prompt):
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=prompt
    )
    return response.text

df=joblib.load("embeddings.joblib")

incoming_query=input("Ask a Question: \n")
question_embeddings=create_embeddings([incoming_query])[0]
similarities=cosine_similarity(np.vstack(df["embedding"]),[question_embeddings]).flatten()
top_results=5
# print(similarities)
max_indx=similarities.argsort()[::-1][0:top_results]
# print(max_indx)
# print(np.vstack(df["embedding"].values))
# print(np.vstack(df["embedding"]).shape)
new_df=df.loc[max_indx]

prompt=f''' I am teaching python in my  100 days of code challenge course. Here are video subtitle chunks containing video title, video number, the start time in seconds, the end time in seconds, and the text at that time:

{new_df[["title","number","start","end","text"]].to_json(orient="records")}
--------------------------

{incoming_query}
user asked this question related to the chunks, you have to answer in a human way (dont add extra unnecessary details like thinking or anything extra from yourside also dont add above content its just for you) where and how much content is taught in which video also at what timestamp in minutes and guide the user to go and watch the particular video. If the user asks unrelated question, tell him that you can answer only the questions related to this course '''

# print(new_df[["title","number","text"]])
with open("prompt.txt","w",encoding="utf-8") as f:
    f.write(prompt)

response=inference_genai(prompt)
with open("response.txt", "w",encoding="utf-8") as f:
    f.write(response)
# for index,item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])