import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Tuple
import markdown2
import chromadb 
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Literal 

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

# --- Load Environment Variables ---
load_dotenv()
MODELS = ["openai/gpt-oss-120b","openai/gpt-oss-20b","meta-llama/llama-4-scout-17b-16e-instruct","llama-3.1-8b-instant", "llama-3.3-70b-versatile"]

GROQ_API_KEY = "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" 
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# --- Paths & Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "db", "chroma_db")
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5" 

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

llm = ChatGroq(model_name="openai/gpt-oss-120b", groq_api_key=GROQ_API_KEY, temperature=0.0)
session_histories: Dict[str, List[Tuple[str, str]]] = {}
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=DB_DIR)

@app.on_event("startup")
def startup_event():
    print("--- Application starting up: Checking for new documents to ingest... ---")
    
    existing_collections = {collection.name for collection in chroma_client.list_collections()}
    print(f"Existing collections in DB: {existing_collections}")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at {DATA_DIR}")
        return

    for doc_file in os.listdir(DATA_DIR):
        if doc_file.endswith(".txt"):
            collection_name = os.path.splitext(doc_file)[0].lower()
            
            if collection_name not in existing_collections:
                print(f"[INGESTING]: New document found: '{doc_file}'. Creating collection '{collection_name}'...")
                
                # 1. Load the document
                doc_path = os.path.join(DATA_DIR, doc_file)
                loader = TextLoader(doc_path, encoding='utf-8')
                documents = loader.load()
                
                # 2. Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                
                # 3. Create the collection in ChromaDB
                Chroma.from_documents(
                    client=chroma_client,
                    documents=chunks,
                    embedding=embeddings,
                    collection_name=collection_name
                )
                print(f"✅ [SUCCESS]: Ingestion complete for '{collection_name}'.")
    print("--- Startup ingestion check complete. ---")


# ---  Prompt for Condensing History ---
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the Cre8AI Assistant. 
    CRITICAL RULES:
    1. Answer to the point and in friendly tone.
    2. No long descriptions or fluff.
    3. Use a friendly tone.
    4. Use the context below for facts. If not found, say 'I'm not sure, please contact our team.'
    5. Respond in {language_name}.
    
    CONTEXT:
    {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

class ChatRequest(BaseModel):
    query: str
    session_id: str
    client_id: str
    language: Literal['en', 'ar'] = 'en'
class ChatResponse(BaseModel):
    answer: str
    session_id: str


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get_chat_ui(request: Request):
    """Serves the main chat UI and dynamically finds available clients."""
    client_names = []
    try:
        collections = chroma_client.list_collections()
        client_names = [
            {"id": col.name, "name": col.name.replace('_', ' ').title()} 
            for col in collections
        ]
    except Exception as e:
        print(f"Error loading client collections: {e}")
    return templates.TemplateResponse("index.html", {"request": request, "clients": client_names})
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

@app.post("/chat")
async def handle_chat_message(request: ChatRequest):
    KNOWLEDGE_BASE = "cre8ai" 
    
    try:
        vectorstore = Chroma(client=chroma_client, collection_name=KNOWLEDGE_BASE, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 10})
    except:
        raise HTTPException(status_code=404, detail="Knowledge base not found.")

    history = [msg for pair in session_histories.get(request.session_id, []) for msg in [HumanMessage(content=pair[0]), AIMessage(content=pair[1])]]
    language_name = "Arabic" if request.language == "ar" else "English"

    last_error = ""
    for model_name in MODELS:
        try:
            llm = ChatGroq(model_name=model_name, groq_api_key=GROQ_API_KEY, temperature=0.1)
            
            retrieved_docs = retriever.invoke(request.query)
            context_text = format_docs(retrieved_docs)

            chain = ANSWER_PROMPT | llm | StrOutputParser()

            answer = chain.invoke({
                "question": request.query,
                "chat_history": history,
                "context": context_text,
                "language_name": language_name
            })

            session_histories.setdefault(request.session_id, []).append((request.query, answer))
            return {"answer": markdown2.markdown(answer), "session_id": request.session_id}

        except Exception as e:
            print(f"DEBUG: Model {model_name} failed: {e}")
            last_error = str(e)
            continue 
            
    raise HTTPException(status_code=500, detail=f"All models failed. Last error: {last_error}")
