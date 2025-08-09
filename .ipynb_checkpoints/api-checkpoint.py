from fastapi import FastAPI
from pydantic import BaseModel
from chatbot_rag import SimpleVectorStore, rag_ask

app = FastAPI()

# Initialize the vector store when the app starts
print("Initializing vector store...")
store = SimpleVectorStore()
store.add_documents("uiuc_complete_content_test.csv")
print("Vector store ready!")

class QueryRequest(BaseModel):
    question: str
    chat_history: list[list[str]] = []

@app.get("/")
def root():
    return {"message": "RAG Chatbot API is running!"}

@app.post("/query")
def query(req: QueryRequest):
    try:
        result = rag_ask(req.question, store)
        return {"response": result}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}