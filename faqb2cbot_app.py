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
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import os, threading, logging, platform, json, time, requests

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
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
EMAIL_FROM = os.getenv("EMAIL_FROM", "noreply@myaitoolset.com")

# Widget customization
WELCOME_MSG = os.getenv("WELCOME_MSG", "Questions? Chat with us!")
THEME_COLOR = os.getenv("THEME_COLOR", "#3B82F6")
SHOW_BRANDING = os.getenv("SHOW_BRANDING", "true").lower() == "true"

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

app.mount("/static", StaticFiles(directory="static"), name="static")

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
    """Inject environment-based variables into widget UI dynamically."""
    path = "widget.html"
    if not os.path.exists(path):
        return PlainTextResponse("widget.html not found", status_code=404)
    html = open(path, encoding="utf-8").read()
    html = (
        html.replace("{{WELCOME_MSG}}", WELCOME_MSG)
            .replace("{{THEME_COLOR}}", THEME_COLOR)
            .replace("{{SHOW_BRANDING}}", "true" if SHOW_BRANDING else "false")
    )
    return HTMLResponse(html)

@app.get("/healthz")
async def healthz(): return {"ok": True, "ts": time.time()}

@app.get("/readyz")
async def readyz(): return {"ready": bool(getattr(app.state, "ready", False))}

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
            "SHOW_BRANDING": SHOW_BRANDING,
            "WELCOME_MSG": WELCOME_MSG,
            "THEME_COLOR": THEME_COLOR,
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
            log.warning("OPENAI_API_KEY missing; skipping pipeline init.")
            return
        embeddings = OpenAIEmbeddings()
        vectorstore = None
        if os.path.isdir(INDEX_DIR):
            try:
                vectorstore = FAISS.load_local(
                    INDEX_DIR, embeddings, allow_dangerous_deserialization=True
                )
                log.info("Loaded FAISS index from disk")
            except Exception as e:
                log.warning(f"Failed loading FAISS: {e}")
        if vectorstore is None:
            if not os.path.exists(FAQ_FILE):
                raise FileNotFoundError(f"{FAQ_FILE} not found")
            log.info("Building FAISS index from faq.txt")
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

# ---------- Lead Email Helper ----------
def send_lead_email(name, email, message):
    """Send chatbot lead via SendGrid"""
    try:
        content = f"Name: {name}\nEmail: {email}\nMessage:\n{message}"
        mail = Mail(
            from_email=EMAIL_FROM,
            to_emails=EMAIL_TO,
            subject=f"New Lead from {name}",
            plain_text_content=content,
        )
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(mail)
        log.info(f"Lead email sent: {response.status_code}")
    except Exception as e:
        log.exception(f"SendGrid send failed: {e}")

# ---------- API ----------
class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Question):
    ensure_pipeline()
    if not app.state.ready:
        raise HTTPException(status_code=503, detail="Initializing, try again soon")
    query_raw = (q.question or "").strip().lower()
    if any(w in query_raw for w in ["book", "appointment", "schedule", "consult"]):
        return JSONResponse({
            "answer": f"You can schedule your appointment here: <a href='{BOOKING_URL}' target='_blank'>{BOOKING_URL}</a>",
            "type": "system"
        })
    if any(w in query_raw for w in ["contact", "email", "reach", "call you", "talk to someone"]):
        return JSONResponse({
            "answer": f"You can contact us anytime at <a href='mailto:{EMAIL_TO}'>{EMAIL_TO}</a>.",
            "type": "system"
        })
    qa = app.state.pipeline["qa"]
    try:
        result = qa.invoke({"query": query_raw})
        answer = result["result"] if isinstance(result, dict) else result
        if not answer.strip():
            answer = "I’m not sure about that — would you like to schedule a quick call to discuss?"
        return JSONResponse({"answer": answer})
    except Exception as e:
        log.exception(f"/ask failed: {e}")
        raise HTTPException(status_code=500, detail="LLM error")
