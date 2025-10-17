# faqb2cbot_app.py — MyAiToolset Enterprise Chatbot (Dynamic Tier + Local Time)
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

@app.get("/healthz") async def healthz(): return {"ok": True, "ts": time.time()}
@app.get("/readyz") async def readyz(): return {"ready": bool(getattr(app.state, "ready", False))}

# ------------------------------------------------------
# core/pipeline.py — FAISS + LLM
# ------------------------------------------------------
app.state.ready = False
app.state.pipeline = None
app.state.lock = threading.Lock()

def build_or_load_pipeline():
    try:
        if not OPENAI_API_KEY:
            log.warning("OPENAI_API_KEY missing; skipping pipeline init."); return
        embeddings = OpenAIEmbeddings()
        vectorstore = None
        if os.path.isdir(INDEX_DIR):
            try:
                vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
                log.info("Loaded FAISS index")
            except Exception as e: log.warning(f"FAISS load failed: {e}")
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

@app.on_event("startup") def warm_bg(): threading.Thread(target=build_or_load_pipeline, daemon=True).start()

# ------------------------------------------------------
# core/emailer.py
# ------------------------------------------------------
def send_lead_email(name, email, phone, message):
    try:
        if not SENDGRID_API_KEY or not EMAIL_TO:
            log.warning("Missing SendGrid config; skipping email."); return
        content = f"Name: {name}\nEmail: {email}\nPhone: {phone}\n\nMessage:\n{message}"
        mail = Mail(from_email=EMAIL_FROM, to_emails=EMAIL_TO,
                    subject=f"New Lead from {name}", plain_text_content=content)
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(mail)
    except Exception as e: log.exception(f"SendGrid send failed: {e}")

# ------------------------------------------------------
# models/schemas.py
# ------------------------------------------------------
class Question(BaseModel): question: str
class Lead(BaseModel):
    name: str; email: str; phone: str; message: str

# ------------------------------------------------------
# core/router.py — Dynamic Tier + Local Time
# ------------------------------------------------------
SMALLTALK = {"hi":"Hey there! How can I help you today?","hello":"Hello! How can I assist you?",
             "thanks":"You're very welcome!","bye":"Take care and have a great day!"}

SYNONYM_MAP = {
    r"\b(prices?|pricing|cost|rates?|fee|charge)\b":"pricing",
    r"\b(hours?|time|open|close|opening|closing)\b":"hours",
    r"\b(location|address|where|find)\b":"location",
    r"\b(book|schedule|appointment|consult|quote|demo)\b":"appointment"
}
FALLBACK_MSG = f"I’m not certain about that — would you like to schedule a chat? {BOOKING_URL}"

INTENT_EXAMPLES = {
    "sales_quote":["I want a quote","how much is it","what do you charge",
                   "send me a pricing estimate","get in touch with sales"],
    "pricing_page":["where is the pricing section","do you have a pricing page",
                    "can I view your plans","is there a pricing tab on website"],
    "faq":["tell me about your services","what do you offer","general question","info about company"]
}

def build_intent_index():
    try:
        emb = OpenAIEmbeddings(); texts=[]; labels=[]
        for lbl,exs in INTENT_EXAMPLES.items():
            for ex in exs: texts.append(ex); labels.append(lbl)
        return FAISS.from_texts(texts, emb, metadatas=[{"intent": l} for l in labels])
    except Exception as e: log.warning(f"Intent index build failed: {e}"); return None

app.state.intent_index = build_intent_index()

def detect_intent(q:str)->str:
    try:
        if not app.state.intent_index: return "faq"
        hit = app.state.intent_index.similarity_search(q,k=1)[0]
        return hit.metadata["intent"]
    except Exception: return "faq"

# --- Tier Features ---
TIER_FEATURES = {
    "business": {"lead": False, "widget": False, "custom": False},
    "elite": {"lead": False, "widget": False, "custom": False},
    "mvp": {"lead": True, "widget": True, "custom": False},
    "business now": {"lead": True, "widget": True, "custom": True},
}

def get_local_time_str():
    try:
        local_tz = get_localzone()
        now = datetime.datetime.now(local_tz)
        return now.strftime("%I:%M %p %Z")
    except Exception:
        return datetime.datetime.utcnow().strftime("%I:%M %p UTC")

def apply_custom_logic_if_enabled(intent,q):
    tier = PLAN_TIER.strip().lower()
    if tier=="business now" and TIER_FEATURES[tier]["custom"]:
        if "promotion" in q:
            return {"type":"qa","answer":"Our current promotion was updated today."}
    return None

def route_intent(q:str):
    q=q.lower().strip()
    for k,r in SMALLTALK.items():
        if q==k or q.startswith(k+" "): return {"type":"smalltalk","answer":r}

    if "time" in q and not any(x in q for x in ["hours","open","close"]):
        return {"type":"utility","answer":f"The current local time is {get_local_time_str()}."}
    if "date" in q:
        return {"type":"utility","answer":f"Today is {datetime.datetime.now().strftime('%A, %B %d, %Y')}"}

    for pat,repl in SYNONYM_MAP.items(): q=re.sub(pat,repl,q)
    intent = detect_intent(q)
    custom = apply_custom_logic_if_enabled(intent,q)
    if custom: return custom

    tier = PLAN_TIER.strip().lower()
    cfg = TIER_FEATURES.get(tier,TIER_FEATURES["business"])

    if intent=="sales_quote":
        if cfg["lead"]:
            msg = "I'd be happy to arrange a callback — please share your name, email, and phone."
            return {"type":"lead","answer":msg}
        else:
            return {"type":"system","answer":"Our pricing details are available online — no contact form needed."}

    if intent=="pricing_page": return {"type":"qa","query":q}
    return {"type":"qa","query":q}

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

    route = route_intent(user_input)
    if route["type"] != "qa":
        return JSONResponse({"answer": route["answer"], "type": route["type"]})

    qa = app.state.pipeline["qa"]
    try:
        system_prompt = ("You are a helpful, professional virtual assistant for MyAiToolset. "
                         "Always sound confident and polite. Never say 'I don't know'. "
                         "If unsure, suggest connecting the user with our team.")
        query_payload = f"{system_prompt}\n\nUser: {route['query']}\nAssistant:"
        result = qa.invoke({"query": query_payload})
        answer = result["result"] if isinstance(result, dict) else result
        if not answer.strip() or "I don't know" in answer:
            answer = FALLBACK_MSG
        return JSONResponse({"answer": answer})
    except Exception as e:
        log.exception(f"/ask failed: {e}")
        return JSONResponse({"answer": FALLBACK_MSG, "type": "error"})
