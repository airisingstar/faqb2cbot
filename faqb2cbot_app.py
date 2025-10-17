# faqb2cbot_app.py — MyAiToolset Enterprise Chatbot (Lead Lock Mode for MVP + Now)
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
import os, threading, logging, time, datetime, re
from tzlocal import get_localzone

# ------------------------------------------------------
# core/config.py
# ------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("faqb2cbot")
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
WELCOME_MSG = os.getenv("WELCOME_MSG", "Questions? Chat with us!")
THEME_COLOR = os.getenv("THEME_COLOR", "#3B82F6")
SHOW_BRANDING = os.getenv("SHOW_BRANDING", "true").lower() == "true"

allow_origins_list = ["*"] if ALLOW_ORIGINS.strip() == "*" else [
    o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()
]

# ------------------------------------------------------
# app/main.py
# ------------------------------------------------------
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
    return HTMLResponse("<h3>MyAiToolset Chatbot is live</h3><p>See /widget.html</p>")

@app.get("/widget.html", response_class=HTMLResponse)
async def get_widget():
    path = "widget.html"
    if not os.path.exists(path):
        return PlainTextResponse("widget.html not found", status_code=404)
    html = open(path, encoding="utf-8").read()
    html = html.replace("{{WELCOME_MSG}}", WELCOME_MSG)\
               .replace("{{THEME_COLOR}}", THEME_COLOR)\
               .replace("{{SHOW_BRANDING}}", "true" if SHOW_BRANDING else "false")
    return HTMLResponse(html)

@app.get("/healthz")
async def healthz():
    return {"ok": True, "ts": time.time()}

@app.get("/readyz")
async def readyz():
    return {"ready": bool(getattr(app.state, "ready", False))}

# ------------------------------------------------------
# core/pipeline.py
# ------------------------------------------------------
app.state.ready = False
app.state.pipeline = None
app.state.lock = threading.Lock()

def build_or_load_pipeline():
    try:
        if not OPENAI_API_KEY:
            log.warning("OPENAI_API_KEY missing; skipping pipeline init.")
            return
        embeddings = OpenAIEmbeddings()
        vectorstore = None
        if os.path.isdir(INDEX_DIR):
            try:
                vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
                log.info("Loaded FAISS index")
            except Exception as e:
                log.warning(f"FAISS load failed: {e}")
        if vectorstore is None:
            loader = TextLoader(FAQ_FILE, encoding="utf-8")
            docs = loader.load()
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            chunks = splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            os.makedirs(INDEX_DIR, exist_ok=True)
            vectorstore.save_local(INDEX_DIR)
        llm = ChatOpenAI(temperature=0, model=OPENAI_MODEL)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=False)
        app.state.pipeline = {"qa": qa}
        app.state.ready = True
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
def warm_bg():
    threading.Thread(target=build_or_load_pipeline, daemon=True).start()

# ------------------------------------------------------
# core/emailer.py
# ------------------------------------------------------
def send_lead_email(name, email, phone, message):
    try:
        if not SENDGRID_API_KEY or not EMAIL_TO:
            log.warning("Missing SendGrid config; skipping email.")
            return
        content = f"Name: {name}\nEmail: {email}\nPhone: {phone}\n\nMessage:\n{message}"
        mail = Mail(from_email=EMAIL_FROM, to_emails=EMAIL_TO,
                    subject=f"New Lead from {name}", plain_text_content=content)
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        resp = sg.send(mail)
        log.info(f"Lead email response: {resp.status_code}")
    except Exception as e:
        log.exception(f"SendGrid send failed: {e}")

# ------------------------------------------------------
# models/schemas.py
# ------------------------------------------------------
class Question(BaseModel):
    question: str

class Lead(BaseModel):
    name: str
    email: str
    phone: str
    message: str

# ------------------------------------------------------
# core/router.py — Lead Lock Mode
# ------------------------------------------------------
LEAD_KEYWORDS = [
    "buy", "book", "appointment", "quote", "pricing", "price", "cost", "estimate",
    "talk to agent", "live agent", "speak to rep", "call", "schedule", "contact",
    "promo", "discount", "offer", "special"
]

TIER_FEATURES = {
    "business": {"lead": False},
    "elite": {"lead": False},
    "mvp": {"lead": True},
    "business now": {"lead": True},
}

FALLBACK_MSG = f"I’m not certain about that — would you like to schedule a chat? {BOOKING_URL}"

# ------------------------------------------------------
# routes
# ------------------------------------------------------
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

    # Determine plan tier
    tier = PLAN_TIER.strip().lower()
    features = TIER_FEATURES.get(tier, TIER_FEATURES["business"])

    # ------------------------------------------------------
    # LEAD LOCK MODE for MVP + NOW
    # ------------------------------------------------------
    if features["lead"]:
        if any(word in user_input.lower() for word in LEAD_KEYWORDS):
            log.info(f"Lead intent detected for tier {tier}: {user_input}")
            return JSONResponse({
                "type": "lead_form_request",
                "answer": (
                    "Let's get your request started! Please confirm your contact details below "
                    "so our team can reach out promptly."
                ),
                "fields": ["name", "email", "phone", "message"]
            })

    # ------------------------------------------------------
    # Default retrieval flow
    # ------------------------------------------------------
    qa = app.state.pipeline["qa"]
    try:
        system_prompt = (
            "You are a helpful, professional virtual assistant for MyAiToolset. "
            "Always sound confident and polite. Never say 'I don't know'. "
            "If unsure, suggest connecting the user with our team."
        )
        query_payload = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
        result = qa.invoke({"query": query_payload})
        answer = result["result"] if isinstance(result, dict) else result
        if not answer.strip() or "I don't know" in answer:
            answer = FALLBACK_MSG
        return JSONResponse({"answer": answer, "type": "qa"})
    except Exception as e:
        log.exception(f"/ask failed: {e}")
        return JSONResponse({"answer": FALLBACK_MSG, "type": "error"})
