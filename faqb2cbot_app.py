from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()

# Initialize app
app = FastAPI()

# Enable CORS to allow frontend access to /ask
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"] to restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve widget and other static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/widget.html", response_class=HTMLResponse)
async def get_widget():
    return FileResponse("widget.html")

# Optional: blank favicon to avoid 404 errors
@app.get("/favicon.ico")
async def favicon():
    return HTMLResponse(content="", status_code=204)


# Load and embed the document
loader = TextLoader("faq.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff")


# Question input schema
class Question(BaseModel):
    question: str


# Chatbot API route
@app.post("/ask")
async def ask_question(q: Question):
    query = q.question
    docs = vectorstore.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    return JSONResponse(content={"answer": response})
