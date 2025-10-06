from fastapi import FastAPI, HTTPException, Response, Form
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
import os, threading, logging, platform, json, time, smtplib
from email.mime.text import MIMEText

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("faqb2cbot")

# ---------- Config ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
FAQ_FILE = os.getenv("FAQ_FILE", "faq.txt")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BOOKING_URL = os.getenv("BOOKING_URL", "https://calendly.com/myaitoolset/15min")
EMAIL_TO = os.getenv("EMAIL_TO", "info@myaitoolset.com")

allow_origins_list = ["*"] if ALLOW_ORIGINS.strip() == "*" else [o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()]

# ---------- App ----------
app = FastAPI(title="MyAiToolset Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse("<h3>MyAiToolset Chatbot is live</h3><p>Try <code>/widget.html</code>, <code>/healthz</code>, or <code>/diag</code></p>")

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

@app.get("/readyz")
async def readyz():
    return {"ready": bool(getattr(app.state, "ready", False))}

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
        },
        "state": {
            "ready": bool(getattr(app.state, "ready", False)),
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
        embeddings = OpenAIEmbeddings()
        vectorstore = None
        if os.path.isdir(INDEX_DIR):
            try:
                vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
                log.info("Loaded FAISS index from disk")
            except Exception as e:
                log.warning(f"Failed loading FAISS: {e}")
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

        llm = ChatOpenAI(temperature=0, model=OPENAI_MODEL)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=False,
        )

        app.state.pipeline = {"qa": qa}
        app.state.ready = True
        log.info("Pipeline ready")
    except Exception as e:
        log.exception(f"Pipeline init failed: {e}")

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
        raise HTTPException(status_code=503, detail="Initializing, try again soon")

    query = (q.question or "").strip().lower()
    qa = app.state.pipeline["qa"]

    # Booking and lead detection
    if any(word in query for word in ["book", "appointment", "schedule", "meeting"]):
        return JSONResponse({
            "answer": f"You can easily book a call here: <a href='{BOOKING_URL}' target='_blank'>{BOOKING_URL}</a>"
        })

    if any(word in query for word in ["contact", "email", "reach", "call you", "talk to someone"]):
        return JSONResponse({
            "answer": f"You can contact us anytime at <a href='mailto:{EMAIL_TO}'>{EMAIL_TO}</a>."
        })

    try:
        result = qa.invoke({"query": query})
        answer = result["result"] if isinstance(result, dict) and "result" in result else result
        return JSONResponse({"answer": answer})
    except Exception as e:
        log.exception(f"/ask failed: {e}")
        raise HTTPException(status_code=500, detail="LLM error")

# ---------- Lead Capture ----------
@app.post("/lead")
async def capture_lead(name: str = Form(...), email: str = Form(...), message: str = Form("")):
    lead = {
        "name": name,
        "email": email,
        "message": message,
        "timestamp": time.time()
    }
    os.makedirs("leads", exist_ok=True)
    with open(f"leads/{int(time.time())}.json", "w") as f:
        json.dump(lead, f, indent=2)

    # Try email notify
    try:
        body = f"New lead captured:\n\nName: {name}\nEmail: {email}\nMessage: {message}"
        msg = MIMEText(body)
        msg["Subject"] = "New Lead - MyAiToolset"
        msg["From"] = EMAIL_TO
        msg["To"] = EMAIL_TO
        with smtplib.SMTP("localhost") as s:
            s.send_message(msg)
    except Exception as e:
        log.warning(f"Email notification failed: {e}")

    return JSONResponse({"ok": True, "saved": True})
