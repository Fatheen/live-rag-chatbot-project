import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_QUERY = os.getenv("API_QUERY", "http://127.0.0.1:8000/query")
API_LIVE  = os.getenv("API_LIVE",  "http://127.0.0.1:8000/ask_live")

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")
st.title("RAG Chatbot 💬")

# ---------------- Session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []   
if "mode" not in st.session_state:
    st.session_state.mode = "Live"  
if "site_root" not in st.session_state:
    st.session_state.site_root = "https://www.busfin.uillinois.edu"

# ---------------- Chat history display ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
      
        if msg["role"] == "assistant" and msg.get("source"):
            st.markdown(
                f"<div style='text-align:right; color:#9aa0a6; font-style:italic;'>Source: {msg['source']}</div>",
                unsafe_allow_html=True
            )

# ---------------- Bottom toolbar (always just above chat input) ----------------
toolbar = st.container()
with toolbar:
    cols = st.columns([1, 2.5]) 
    with cols[0]:
      
        mode = st.radio(
            "Answer source",
            ["CSV", "Live"],
            index=(0 if st.session_state.mode == "CSV" else 1),
            horizontal=True,
            label_visibility="collapsed",
        )
    with cols[1]:
       
        sr = st.text_input(
            "Site root (Live mode)",
            value=st.session_state.site_root,
            label_visibility="collapsed",
            placeholder="https://www.example.edu",
        )

   
    st.session_state.mode = "CSV" if mode == "CSV" else "Live"
    st.session_state.site_root = sr

# ---------------- Chat input ----------------
prompt = st.chat_input("Ask me anything...")

if prompt:
   
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "source": None})

   
    chat_history = [
        [st.session_state.messages[i]["content"], st.session_state.messages[i + 1]["content"]]
        for i in range(0, len(st.session_state.messages) - 1, 2)
        if st.session_state.messages[i]["role"] == "user"
        and i + 1 < len(st.session_state.messages)
        and st.session_state.messages[i + 1]["role"] == "assistant"
    ]

    try:
        if st.session_state.mode == "Live":
            payload = {
                "question": prompt,
                "site": st.session_state.site_root,
                "max_pages": 8,
                "k": 6,
            }
            r = requests.post(API_LIVE, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            reply   = data.get("answer", "")
            sources = data.get("sources", [])

         
            with st.chat_message("assistant"):
                st.write(reply)
               
                if sources and "Sources:" not in (reply or ""):
                    st.markdown("**Sources:**")
                    for u in sources:
                        st.markdown(f"- [{u}]({u})")
                st.markdown(
                    "<div style='text-align:right; color:#9aa0a6; font-style:italic;'>Source: Live</div>",
                    unsafe_allow_html=True,
                )

            st.session_state.messages.append({"role": "assistant", "content": reply, "source": "Live"})

        else: 
            payload = {"question": prompt, "chat_history": chat_history}
            r = requests.post(API_QUERY, json=payload, timeout=30)
            r.raise_for_status()
            reply = r.json().get("response", "")

            with st.chat_message("assistant"):
                st.write(reply)
                st.markdown(
                    "<div style='text-align:right; color:#9aa0a6; font-style:italic;'>Source: CSV</div>",
                    unsafe_allow_html=True,
                )

            st.session_state.messages.append({"role": "assistant", "content": reply, "source": "CSV"})

    except requests.exceptions.RequestException as e:
        err = f"Backend request failed: {e}"
        st.chat_message("assistant").write(err)
        st.session_state.messages.append({"role": "assistant", "content": err, "source": None})
