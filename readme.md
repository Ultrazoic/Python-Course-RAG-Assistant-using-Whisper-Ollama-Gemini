
This project implements a Retrieval Augmented Generation (RAG)
assistant that answers questions about a Python programming course.

The system extracts subtitles from course videos using Whisper,
creates embeddings using Ollama, retrieves relevant transcript
chunks using cosine similarity, and generates answers using
Google Gemini.
# How To use this RAG teaching assistant for your data

## Step 1 - Collect your videos
move your videos into the videos folder

## Step 2 - COnvert to mp3
convert your videos to mp3 by running videos_to_mp3

## Step 3 - Convert to json
convert your mp3 to json by running mp3_to_jsons (use merge_chunks for merging no of chunks together to get better results)

## Step 4 - Covert the json files into vectors
Use the preprocessing_json to create a dataframe of embeddings for all the json files and save it to a joblib pickle

## Step 5 - Prompt Generation and feeding the LLM
Read the joblib file and load it into the memory. Generate a prompt according to the user query and feed it to the LLM to get a appropriate response
