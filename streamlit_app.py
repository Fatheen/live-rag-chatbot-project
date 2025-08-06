import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000/query")

st.title("RAG Chatbot 💬")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("Ask me anything...")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    chat_history = [
        [st.session_state.messages[i]["content"], st.session_state.messages[i+1]["content"]]
        for i in range(0, len(st.session_state.messages)-1, 2)
        if st.session_state.messages[i]["role"] == "user"
    ]

    response = requests.post(API_URL, json={
        "question": prompt,
        "chat_history": chat_history
    }).json()

    reply = response["response"]
    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
