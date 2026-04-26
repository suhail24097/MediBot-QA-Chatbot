# MediBot-QA-Chatbot
MediBot – AI-Based Medical Question Answering System, An AI-powered healthcare chatbot that answers medical queries by combining structured MySQL databases with an LLM-based reasoning engine.
🔹 Project Title

MediBot – AI-Based Medical Question Answering System

🔹 Problem Statement

Explain in 3–4 lines:

An AI-powered healthcare chatbot that answers medical queries by combining structured MySQL databases with an LLM-based reasoning engine.

🔹 Key Features 

Hybrid SQL + LLM query answering

Dynamic intent-based SQL generation

Smart federated querying (disease / medicine DB selection)

LLM-based domain guard (medical-only safety)

Caching for faster responses

🔹 Tech Stack

Python

MySQL

Hugging Face Transformers (Flan-T5)

mysql-connector

LLM-based summarization

🔹 Architecture

(from report):

User query

Domain guard

Intent + entity extraction

SQL query generation

Database retrieval

LLM summarization

Cached response

example query - 
User: What are the symptoms and medicines for diabetes?
System:
- Symptoms: frequent urination, fatigue
- Medicines: Metformin, Insulin


git clone https://github.com/yourusername/MediBot-QA-Chatbot.git
cd MediBot-QA-Chatbot
pip install -r requirements.txt
python backend/app.py
