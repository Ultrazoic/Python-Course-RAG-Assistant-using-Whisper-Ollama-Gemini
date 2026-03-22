# Python Course RAG Assistant (Whisper + Ollama + Gemini)

This project implements a Retrieval-Augmented Generation (RAG) assistant that answers questions about a Python programming course using video transcripts.

The idea of this project is to convert course videos into a searchable knowledge base. The system processes videos, extracts subtitles using Whisper, converts the text into embeddings using Ollama, retrieves the most relevant transcript chunks using semantic search, and generates answers using Google's Gemini model.

The assistant helps users quickly find where a concept is explained in the course and provides the exact video and timestamp to watch.

------------------------------------------------------------

PROJECT ARCHITECTURE

The system follows a complete RAG pipeline:

Course Videos
↓
Audio Extraction (FFmpeg)
↓
Speech-to-Text using Whisper
↓
JSON Transcript Chunks
↓
Chunk Merging
↓
Embedding Generation (Ollama - nomic-embed-text)
↓
Vector Similarity Search (Cosine Similarity)
↓
Prompt Construction
↓
LLM Response using Gemini

------------------------------------------------------------

FEATURES

• Converts course videos into a searchable knowledge base  
• Uses Whisper for speech-to-text transcription  
• Generates embeddings using Ollama (nomic-embed-text)  
• Performs semantic search using cosine similarity  
• Retrieves the most relevant transcript segments  
• Generates human-readable answers using Gemini  
• Guides users to the exact video and timestamp where a concept is taught  

------------------------------------------------------------

TECHNOLOGIES USED

Python  
OpenAI Whisper  
Ollama  
Nomic Embeddings  
Google Gemini API  
Pandas  
NumPy  
Scikit-learn  
FFmpeg  

------------------------------------------------------------

PROJECT WORKFLOW

1. Convert Videos to Audio

Course videos are converted into mp3 files using FFmpeg.

Run:
python videos_to_mp3.py


2. Generate Transcripts using Whisper

Audio files are transcribed using Whisper and converted into JSON subtitle chunks.

Run:
python mp3_to_jsons.py

Each JSON file contains:
- video title
- video number
- start timestamp
- end timestamp
- subtitle text


3. Merge Subtitle Chunks

Small subtitle segments are grouped into larger chunks to improve embedding quality.

Run:
python merge_chunks.py


4. Generate Embeddings

Text chunks are converted into embeddings using the Ollama embedding model.

Run:
python preprocess_json.py

The embeddings are stored in a dataframe for semantic search.


5. Ask Questions

Users can ask questions related to the course.

Run:
python processing_incoming.py

The system will:
1. Convert the user query into embeddings
2. Find the most relevant transcript chunks
3. Construct a prompt using the retrieved content
4. Send the prompt to Gemini
5. Generate an answer with video references

------------------------------------------------------------

EXAMPLE QUERY

Ask a Question:
Where are variables explained?


Example Output:

Variables are explained in Video 6 titled "Variables and Data Types".

Watch from 0:00 to 4:48 for the full explanation.

------------------------------------------------------------

FOLDER STRUCTURE

Python-Course-RAG-Assistant

videos_to_mp3.py
mp3_to_jsons.py
merge_chunks.py
preprocess_json.py
processing_incoming.py

videos/
audios/
jsons/
new_json/

prompt.txt
response.txt
embeddings.joblib

------------------------------------------------------------

INSTALLATION

Clone the repository

git clone https://github.com/Ultrazoic/Python-Course-RAG-Assistant-using-Whisper-Ollama-Gemini

cd Python-Course-RAG-Assistant-using-Whisper-Ollama-Gemini


Install dependencies

pip install -r requirements.txt

You also need to install:
- FFmpeg
- Ollama
- Whisper model

------------------------------------------------------------

REQUIREMENTS

Example requirements.txt

pandas
numpy
scikit-learn
requests
joblib
whisper
google-genai

------------------------------------------------------------

FUTURE IMPROVEMENTS

• Use FAISS or a vector database for faster retrieval  
• Add a web interface using Streamlit  
• Support multiple courses and datasets  
• Improve chunking strategy  
• Build a full AI learning assistant

------------------------------------------------------------

LEARNING OUTCOMES

Through this project I learned:

• Building Retrieval-Augmented Generation (RAG) systems  
• Using Whisper for speech-to-text pipelines  
• Generating embeddings using Ollama  
• Implementing semantic search using cosine similarity  
• Integrating LLMs to generate contextual answers  

------------------------------------------------------------

AUTHOR

Rajpal Choudhary

Aspiring Data Scientist | Machine Learning | AI Projects

GitHub:
https://github.com/Ultrazoic
