# faqb2cbot_app.py
from fastapi import FastAPI, HTTPException, Request
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
BOOKING_URL = os.getenv("BOOKING_URL", "https://formspree.io/f/your_form_id")  # your Formspree link
EMAIL_TO = os.getenv("EMAIL_TO", "info@myaitoolset.com")

allow_origins_list = ["*"] if ALLOW_ORIGINS.strip() == "*" else [
    o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()
]

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


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(
        "<h3>MyAiToolset Chatbot is live</h3>"
        "<p>Try <code>/widget.html</code>, <code>/healthz</code>, or <code>/diag</code></p>"
    )

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
    """Builds or reloads FAISS index. Auto-rebuilds if faq.txt is newer than existing index."""
    log.info("Pipeline init: starting")
    try:
        if not OPENAI_API_KEY:
            log.warning("OPENAI_API_KEY missing; skipping pipeline init.")
            return

        embeddings = OpenAIEmbeddings()
        vectorstore = None

        rebuild = True
        if os.path.isdir(INDEX_DIR):
            try:
                idx_time = os.path.getmtime(INDEX_DIR)
                faq_time = os.path.getmtime(FAQ_FILE)
                if idx_time > faq_time:
                    rebuild = False
            except Exception:
                rebuild = True

        if rebuild:
            log.info("Rebuilding FAISS index from faq.txt")
            loader = TextLoader(FAQ_FILE, encoding="utf-8")
            docs = loader.load()
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            chunks = splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            os.makedirs(INDEX_DIR, exist_ok=True)
            vectorstore.save_local(INDEX_DIR)
            log.info(f"Saved FAISS index to {INDEX_DIR}")
        else:
            log.info("FAQ unchanged — loading existing FAISS index")
            vectorstore = FAISS.load_local(
                INDEX_DIR, embeddings, allow_dangerous_deserialization=True
            )

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
        log.exception(f"Pipeline init failed: {e}")
        app.state.ready = False
        app.state.pipeline = None


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

    query_raw = (q.question or "").strip()
    query = query_raw.lower()

    # Appointment / booking shortcuts
    if any(w in query for w in ["book", "appointment", "schedule", "consult", "reserve"]):
        return JSONResponse({
            "answer": (
                f"You can easily schedule an appointment here: "
                f"<a href='{BOOKING_URL}' target='_blank' rel='noopener'>{BOOKING_URL}</a>"
            ),
            "type": "system"
        })

    # Contact shortcuts
    if any(w in query for w in ["contact", "email", "reach", "call you", "talk to someone"]):
        return JSONResponse({
            "answer": f"You can contact us anytime at <a href='mailto:{EMAIL_TO}'>{EMAIL_TO}</a>.",
            "type": "system"
        })

    # FAQ / RAG
    qa = app.state.pipeline["qa"]
    try:
        result = qa.invoke({"query": query_raw})
        answer = result.get("result") if isinstance(result, dict) else result
        if not answer or len(answer.strip()) < 5:
            answer = (
                "I'm not completely sure about that one, "
                f"but you can ask our staff directly here: "
                f"<a href='{BOOKING_URL}' target='_blank'>Contact Form</a>."
            )
        return JSONResponse({"answer": answer})
    except Exception as e:
        log.exception(f"/ask failed: {e}")
        raise HTTPException(status_code=500, detail="LLM error")


# ---------- Lead Capture ----------
@app.post("/lead")
async def capture_lead(request: Request):
    """
    Accepts:
      - JSON: { name, email?, phone?, preferred_time?, message?, site? }
      - form-encoded or multipart: same field names
    """
    data = {}
    ctype = request.headers.get("content-type", "").lower()
    try:
        if "application/json" in ctype:
            data = await request.json()
        elif "application/x-www-form-urlencoded" in ctype or "multipart/form-data" in ctype:
            form = await request.form()
            data = dict(form)
    except Exception as e:
        log.warning(f"/lead parse failed: {e}")
        data = {}

    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    phone = (data.get("phone") or "").strip()
    preferred_time = (data.get("preferred_time") or "").strip()
    message = (data.get("message") or "").strip()
    site = (data.get("site") or "").strip()

    if not name:
        return JSONResponse({"ok": False, "error": "name required"}, status_code=400)

    payload = {
        "name": name,
        "email": email,
        "phone": phone,
        "preferred_time": preferred_time,
        "message": message,
        "site": site,
        "timestamp": time.time(),
    }

    os.makedirs("leads", exist_ok=True)
    with open(f"leads/{int(time.time())}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    try:
        body = (
            "New lead captured:\n\n"
            f"Name: {name}\n"
            f"Email: {email}\n"
            f"Phone: {phone}\n"
            f"Preferred time: {preferred_time}\n"
            f"Message: {message}\n"
            f"Site: {site}\n"
        )
        msg = MIMEText(body)
        msg["Subject"] = "New Lead - MyAiToolset"
        msg["From"] = EMAIL_TO
        msg["To"] = EMAIL_TO
        with smtplib.SMTP("localhost") as s:
            s.send_message(msg)
    except Exception as e:
        log.warning(f"Email notification failed (expected if SMTP not configured): {e}")

    return JSONResponse({"ok": True, "saved": True})
