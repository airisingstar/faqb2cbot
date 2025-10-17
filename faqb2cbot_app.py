# faqb2cbot_app.py  —  Enterprise-Grade Chatbot for MyAiToolset
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
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
import os, threading, logging, platform, time, datetime, re

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
EMAIL_TO = os.getenv("EMAIL_TO")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
EMAIL_FROM = os.getenv("EMAIL_FROM", "noreply@myaitoolset.com")
PLAN_TIER = os.getenv("PLAN_TIER", "business").lower()

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

# ---------- Basic Routes ----------
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
            "PLAN_TIER": PLAN_TIER,
        },
        "files": {
            "faq_exists": os.path.exists(FAQ_FILE),
            "index_dir_exists": os.path.isdir(INDEX_DIR),
            "widget_exists": os.path.exists("widget.html"),
        },
        "state": {"ready": bool(getattr(app.state, "ready", False))}
    }
    return JSONResponse(info)

# ---------- Lazy Pipeline ----------
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

# ---------- SendGrid Helper ----------
def send_lead_email(name, email, phone, message):
    try:
        if not SENDGRID_API_KEY or not EMAIL_TO:
            log.warning("Missing SendGrid configuration; skipping lead email.")
            return
        content = f"Name: {name}\nEmail: {email}\nPhone: {phone}\n\nMessage:\n{message}"
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

# ---------- Models ----------
class Question(BaseModel):
    question: str

class Lead(BaseModel):
    name: str
    email: str
    phone: str
    message: str

# ---------- Core Logic Enhancements ----------
SMALLTALK = {
    "hi": "Hey there! How can I help you today?",
    "hello": "Hello! How can I assist you?",
    "hey": "Hi there 👋",
    "yo": "Hey! What can I do for you?",
    "thanks": "You're very welcome!",
    "thank you": "You're very welcome!",
    "bye": "Take care and have a great day!",
    "goodbye": "Goodbye! Hope to chat again soon.",
    "help": "Sure, I can help. What would you like to know?",
}

SYNONYM_MAP = {
    r"\b(prices?|pricing|cost|rates?|fee|charge)\b": "pricing",
    r"\b(hours?|time|open|close|opening|closing)\b": "hours",
    r"\b(location|address|where|find)\b": "location",
    r"\b(book|schedule|appointment|consult|quote|demo)\b": "appointment",
}

FALLBACK_MSG = (
    "I’m not certain about that — but I’d be happy to connect you with our team "
    f"or help you schedule a quick chat here: {BOOKING_URL}"
)

# ---------- Intent Router ----------
def route_intent(query: str):
    q = query.lower().strip()
    # Normalize synonyms
    for pattern, replacement in SYNONYM_MAP.items():
        q = re.sub(pattern, replacement, q)

    # Smalltalk / greetings
    for key, resp in SMALLTALK.items():
        if q == key or q.startswith(key + " "):
            return {"type": "smalltalk", "answer": resp}

    # Utility intents
    if "time" in q:
        now = datetime.datetime.now().strftime("%I:%M %p")
        return {"type": "utility", "answer": f"The current time is {now}."}
    if "date" in q:
        today = datetime.datetime.now().strftime("%A, %B %d, %Y")
        return {"type": "utility", "answer": f"Today is {today}."}

    # Booking / Contact intents
    if any(word in q for word in ["appointment", "quote", "demo", "pricing", "contact", "call", "email"]):
        if PLAN_TIER == "business":
            return {"type": "system", "answer": "If you’d like to schedule an appointment or speak to a live representative, please refer to the contact section of the website."}
        else:
            return {"type": "lead", "answer": "Sure! I can help with that. Please provide your name, email, phone number, and a brief message so we can reach out to you."}

    # Default → fallback to QA
    return {"type": "qa", "query": q}

# ---------- API ----------
@app.post("/lead")
async def collect_lead(lead: Lead):
    send_lead_email(lead.name, lead.email, lead.phone, lead.message)
    return {"ok": True, "msg": "Lead sent successfully"}

@app.post("/ask")
async def ask(q: Question):
    ensure_pipeline()
    if not app.state.ready:
        raise HTTPException(status_code=503, detail="Initializing, try again soon")

    user_input = (q.question or "").strip()
    if not user_input:
        return JSONResponse({"answer": "Could you please provide more details?"})

    # Step 1: route by intent
    route = route_intent(user_input)
    if route["type"] != "qa":
        return JSONResponse({"answer": route["answer"], "type": route["type"]})

    # Step 2: run RetrievalQA with brand tone
    qa = app.state.pipeline["qa"]
    try:
        system_prompt = (
            "You are a helpful, professional virtual assistant for MyAiToolset. "
            "Always sound confident and polite. Never say 'I don't know'. "
            "If unsure, suggest connecting the user with our team."
        )
        query_payload = f"{system_prompt}\n\nUser: {route['query']}\nAssistant:"
        result = qa.invoke({"query": query_payload})
        answer = result["result"] if isinstance(result, dict) else result
        answer = answer.strip()
        if not answer or "I don't know" in answer:
            answer = FALLBACK_MSG
        return JSONResponse({"answer": answer})
    except Exception as e:
        log.exception(f"/ask failed: {e}")
        return JSONResponse({"answer": FALLBACK_MSG, "type": "error"})
