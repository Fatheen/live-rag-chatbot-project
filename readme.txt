# RAG Chatbot for Business & Finance Website

This project is a chatbot system that uses **Retrieval-Augmented Generation (RAG)** to answer questions based on content from the University of Illinois Business & Finance website.

It includes:
- A **FastAPI** backend (`api.py`)
- A **retrieval + generation module** (`chatbot_rag.py`)
- A **Streamlit UI** (`streamlit_app.py`)
- A **CSV file** with scraped and cleaned website content


HOW TO RUN:
Run the Backend (FastAPI)
uvicorn api:app --reload --host 0.0.0.0 --port 8000

Run the Frontend (Streamlit)
streamlit run streamlit_app.py

Access the UI
https://jupyter.gpu.atlas.illinois.edu/user/<NetID@illinois.edu>/proxy/8501/?redirects=1

50 test questions:
https://docs.google.com/document/d/1i_jmUYqEdjwQX0v2oM96r2h2Dny4aZDDqqzz7vBJVDA/edit?usp=sharin