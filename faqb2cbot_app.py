from fastapi import FastAPI, HTTPException, Response
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
import os, threading

# ------------------ Config ------------------
load_dotenv()  # local only; on Render use env vars
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
FAQ_FILE = os.getenv("FAQ_FILE", "faq.txt")

allow_origins_list = (
    ["*"] if ALLOW_ORIGINS.strip() == "*"
    else [o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()]
)

# ------------------ App ------------------
app = FastAPI(title="FAQB2CBot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve widget/static
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse("<h3>FAQB2CBot is live</h3><p>Try <code>/widget.html</code>, <code>/healthz</code>, <code>/readyz</code></p>")

# HEAD handlers to avoid 405s from health pings
@app.head("/")
async def head_root():
    return Response(status_code=200)

@app.get("/favicon.ico")
async def favicon():
    return HTMLResponse(content="", status_code=204)

@app.get("/widget.html", response_class=HTMLResponse)
async def get_widget():
    return FileResponse("widget.html")

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.head("/healthz")
async def head_healthz():
    return Response(status_code=200)

@app.get("/readyz")
async def readyz():
    return {"ready": bool(getattr(app.state, "ready", False))}

@app.head("/readyz")
async def head_readyz():
    return Response(status_code=200)

# ------------------ Lazy pipeline ------------------
app.state.ready = False
app.state.pipeline = None
app.state.lock = threading.Lock()

def build_or_load_pipeline():
    embeddings = OpenAIEmbeddings()
    llm = OpenAI(temperature=0)

    vectorstore = None
    if os.path.isdir(INDEX_DIR):
        try:
            vectorstore = FAISS.load_local(
                INDEX_DIR, embeddings, allow_dangerous_deserialization=True
            )
        except Exception:
            vectorstore = None

    if vectorstore is None:
        if not os.path.exists(FAQ_FILE):
            raise FileNotFoundError(f"{FAQ_FILE} not found")
        loader = TextLoader(FAQ_FILE, encoding="utf-8")
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        os.makedirs(INDEX_DIR, exist_ok=True)
        vectorstore.save_local(INDEX_DIR)

    chain = load_qa_chain(llm, chain_type="stuff")
    app.state.pipeline = {"vectorstore": vectorstore, "chain": chain}
    app.state.ready = True

def ensure_pipeline():
    if app.state.pipeline is None:
        with app.state.lock:
            if app.state.pipeline is None:
                build_or_load_pipeline()

@app.on_event("startup")
def warm_in_background():
    threading.Thread(target=build_or_load_pipeline, daemon=True).start()

# ------------------ API ------------------
class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Question):
    ensure_pipeline()
    if not app.state.ready:
        raise HTTPException(status_code=503, detail="Warming up, try again in a moment.")
    query = q.question.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty question")
    vs = app.state.pipeline["vectorstore"]
    chain = app.state.pipeline["chain"]
    docs = vs.similarity_search(query, k=4)
    answer = chain.run(input_documents=docs, question=query)
    return JSONResponse({"answer": answer})
