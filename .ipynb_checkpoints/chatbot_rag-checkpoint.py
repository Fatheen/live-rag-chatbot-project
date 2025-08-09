#!/usr/bin/env python
# coding: utf-8

import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# LLM request
def send(query: str) -> str:
    url = "https://llm.gpu.atlas.illinois.edu/api/generate"
    myobj = {
        "model": "gemma3:12b",
        "prompt": query,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(
            url,
            data=json.dumps(myobj),
            headers=headers,
            auth=('atlasaiteam', 'jx@U2WS8BGSqwu'),
            timeout=1000
        )
        response.raise_for_status()
        data = response.json()
        return data.get('response', '[Error: "response" field not found in LLM output]')
    except requests.exceptions.RequestException as e:
        return f"[Network error: {e}]"
    except json.JSONDecodeError:
        return f"[Error: Invalid JSON returned: {response.text}]"

# vector store
class SimpleVectorStore:
    def __init__(self, model_name: 
                 str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.texts: List[str] = []
        self.embeddings: np.ndarray = None

    # def add_documents(self, source: Union[str, List[str]]):
    #     # if isinstance(source, str):
    #     #     loader = CSVLoader(file_path=source, encoding="utf-8")
    #     #     raw_docs = loader.load()
    #     #     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    #     #     docs: List[str] = [doc.page_content for doc in splitter.split_documents(raw_docs)]
    #     if isinstance(source, str):
    #         # FIX:force CSV loader to treat the first column as content
    #         loader = CSVLoader(file_path=source, encoding="utf-8", csv_args={"delimiter": ","})

    #         raw_docs = loader.load()
            
    #         print(f"[DEBUG] Loaded {len(raw_docs)} raw docs")
    #         print("[DEBUG] First doc preview:", raw_docs[0].page_content[:300])
        
    #         splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)
    #         docs: List[str] = [doc.page_content for doc in splitter.split_documents(raw_docs)]
        
    #         print(f"[DEBUG] After splitting: {len(docs)} chunks")
    #         print("[DEBUG] First chunk:", docs[0][:300])

    #     else:
    #         docs = source

    #     embs = self.embedder.encode(docs, convert_to_numpy=True)
    #     self.embeddings = embs if self.embeddings is None else np.vstack([self.embeddings, embs])
    #     self.texts.extend(docs)
    def add_documents(self, source: Union[str, List[str]]):
        if isinstance(source, str):
        # load only the 'Text' column from your CSV
            df = pd.read_csv(source)
            raw_texts = df["Text"].dropna().tolist()
    
            # more effective chunking
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=175)
    
            chunks = []
            for text in raw_texts:
                chunks.extend(splitter.split_text(text))
    
            print(f"[DEBUG] Loaded {len(chunks)} text chunks.")
            print("[DEBUG] First chunk preview:", chunks[0][:300])
    
        else:
            chunks = source
    
        embs = self.embedder.encode(chunks, convert_to_numpy=True)
        self.embeddings = embs if self.embeddings is None else np.vstack([self.embeddings, embs])
        self.texts.extend(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        top_indices = np.argsort(sims)[-top_k:][::-1]
        st.subheader("🔍 Retrieved Chunks:")
        for i in top_indices:
            st.markdown(f"**Chunk {i}**:\n```\n{self.texts[i][:500]}\n```")
        
        return [self.texts[i] for i in top_indices]

#Rag Q&A
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)

def rag_ask(question: str, store: SimpleVectorStore, top_k: int = 5) -> str:
    docs = store.retrieve(question, top_k)
    context = "\n\n".join(docs)
    history = memory.load_memory_variables({})["chat_history"]
    prompt = (
        "Use the provided context and conversation history to answer the question. Prioritize the provided context as the main source of truth.\n"
        "Do not treat the question itself as context.\n"
        "If the answer is not found in the context, clearly inform the user, then provide an answer based on your own knowledge.\n"
        f"{history}\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    answer = send(prompt)
    memory.save_context({"input": question}, {"output": answer})
    return answer

if __name__ == "__main__":
    store = SimpleVectorStore()
    # store.add_documents([
    #     "Python is a programming language.",
    #     "LangChain enables chaining LLMs with tools.",
    #     "RAG combines retrieval with generation."
    # ])
    store.add_documents("uiuc_complete_content_test.csv")
    print("Chatbot ready. Type your question or 'exit' to quit.")
    while True:
        user_question = input("You: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        response = rag_ask(user_question, store)
        print("Bot:", response)