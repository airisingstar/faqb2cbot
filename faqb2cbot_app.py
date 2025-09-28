from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os

# ---------- Config ----------
load_dotenv()  # optional locally
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Don't crash locally, but Render MUST have this env var set.
    print("WARNING: OPENAI_API_KEY not set. Set it in hosting env.")

# Allow CORS from anywhere for testing; tighten in prod (comma-separated env)
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
allow_origins_list = (
    ["*"] if ALLOW_ORIGINS.strip() == "*"
    else [o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()]
)

# ---------- App ----------
app = FastAPI(title="FAQB2CBot")

# CORS so widget (iframe) can call /ask
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static + widget
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse("<h3>FAQB2CBot is live</h3><p>Try <code>/widget.html</code></p>")

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.get("/widget.html", response_class=HTMLResponse)
async def get_widget():
    return FileResponse("widget.html")

@app.get("/favicon.ico")
async def favicon():
    return HTMLResponse(content="", status_code=204)

# ---------- RAG init (simple on-start load) ----------
# NOTE: For bigger corpora, prebuild FAISS and load from disk.
loader = TextLoader("faq.txt", encoding="utf-8")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff")

# ---------- API ----------
class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Question):
    query = q.question
    docs = vectorstore.similarity_search(query, k=4)
    answer = chain.run(input_documents=docs, question=query)
    return JSONResponse({"answer": answer})
