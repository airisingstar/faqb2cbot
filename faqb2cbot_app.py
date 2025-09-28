from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
import os, threading, logging, platform, json, time

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("faqb2cbot")

# ---------- Config ----------
load_dotenv()  # local only; on Render use env vars
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
FAQ_FILE = os.getenv("FAQ_FILE", "faq.txt")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # adjust if needed

allow_origins_list = ["*"] if ALLOW_ORIGINS.strip() == "*" else [o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()]

# ---------- App ----------
app = FastAPI(title="FAQB2CBot")

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
    return HTMLResponse("<h3>FAQB2CBot is live</h3><p>Try <code>/widget.html</code>, <code>/healthz</code>, <code>/readyz</code>, <code>/diag</code></p>")

@app.head("/")
async def head_root():
    return Response(status_code=200)

@app.get("/favicon.ico")
async def favicon():
    return HTMLResponse(content="", status_code=204)

@app.get("/widget.html", response_class=HTMLResponse)
async def get_widget():
    path = "widget.html"
    if not os.path.exists(path):
        return PlainTextResponse("widget.html not found", status_code=404)
    return FileResponse(path)

@app.get("/healthz")
async def healthz():
    return {"ok": True, "ts": time.time()}

@app.head("/healthz")
async def head_healthz():
    return Response(status_code=200)

@app.get("/readyz")
async def readyz():
    return {"ready": bool(getattr(app.state, "ready", False))}

@app.head("/readyz")
async def head_readyz():
    return Response(status_code=200)

@app.get("/diag")
async def diag():
    info = {
        "python": platform.python_version(),
        "cwd": os.getcwd(),
        "env": {
            "ALLOW_ORIGINS": ALLOW_ORIGINS,
            "INDEX_DIR": INDEX_DIR,
            "FAQ_FILE": FAQ_FILE,
            "OPENAI_MODEL": OPENAI_MODEL,
            "OPENAI_API_KEY_set": bool(OPENAI_API_KEY),
        },
        "files": {
            "faq_exists": os.path.exists(FAQ_FILE),
            "index_dir_exists": os.path.isdir(INDEX_DIR),
            "widget_exists": os.path.exists("widget.html"),
        },
        "state": {
            "ready": bool(getattr(app.state, "ready", False)),
            "pipeline_built": app.state.__dict__.get("pipeline") is not None,
        }
    }
    return JSONResponse(info)

# ---------- Lazy pipeline ----------
app.state.ready = False
app.state.pipeline = None
app.state.lock = threading.Lock()

def build_or_load_pipeline():
    log.info("Pipeline init: starting")
    try:
        if not OPENAI_API_KEY:
            log.warning("OPENAI_API_KEY is not set. LLM will fail.")
        embeddings = OpenAIEmbeddings()

        # Load FAISS from disk if present
        vectorstore = None
        if os.path.isdir(INDEX_DIR):
            log.info(f"Trying to load FAISS index from {INDEX_DIR}")
            try:
                vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
                log.info("Loaded FAISS index from disk")
            except Exception as e:
                log.warning(f"Failed loading FAISS from disk: {e}")
                vectorstore = None

        # Otherwise build from faq.txt
        if vectorstore is None:
            if not os.path.exists(FAQ_FILE):
                raise FileNotFoundError(f"{FAQ_FILE} not found")
            log.info(f"Building FAISS from {FAQ_FILE}")
            loader = TextLoader(FAQ_FILE, encoding="utf-8")
            docs = loader.load()
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            chunks = splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            os.makedirs(INDEX_DIR, exist_ok=True)
            vectorstore.save_local(INDEX_DIR)
            log.info(f"Saved FAISS index to {INDEX_DIR}")

        llm = ChatOpenAI(temperature=0, model=OPENAI_MODEL)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=False,
        )

        app.state.pipeline = {"qa": qa}
        app.state.ready = True
        log.info("Pipeline init: success; ready=True")

    except Exception as e:
        app.state.pipeline = None
        app.state.ready = False
        log.exception(f"Pipeline init: FAILED: {e}")

def ensure_pipeline():
    if app.state.pipeline is None:
        with app.state.lock:
            if app.state.pipeline is None:
                build_or_load_pipeline()

@app.on_event("startup")
def warm_in_background():
    threading.Thread(target=build_or_load_pipeline, daemon=True).start()

# ---------- API ----------
class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Question):
    ensure_pipeline()
    if not app.state.ready:
        raise HTTPException(status_code=503, detail="Warming up or failed to init; check /diag")

    query = (q.question or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty question")

    qa = app.state.pipeline["qa"]
    try:
        result = qa.invoke({"query": query})
        # RetrievalQA returns dict with 'result'
        answer = result["result"] if isinstance(result, dict) and "result" in result else result
        return JSONResponse({"answer": answer})
    except Exception as e:
        log.exception(f"/ask failed: {e}")
        raise HTTPException(status_code=500, detail="LLM error")

