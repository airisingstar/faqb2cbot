# 🧠 AiToolset — FastAPI RAG Service (Render Deployment)

**Repository:** `https://github.com/airisingstar/faqb2cbot/tree/main`  
**Primary module:** `faqb2cbot_app.py`  
**Deployed service name:** `faqb2cbot-api`  
**Frontend domain:** [https://myaitoolset.com](https://myaitoolset.com)

---

## 🧩 Overview

**AiToolset** provides a Retrieval-Augmented Generation (RAG) backend API that powers the embedded chatbot for websites.

It combines **FastAPI + LangChain + FAISS** with **OpenAI APIs** for natural Q&A, and integrates with **WordPress** frontends and **Formspree** for contact intake.

---

## ⚙️ Architecture

### High-Level Flow
```
User Browser
   │
   ▼
WordPress (myaitoolset.com)
 ├─ Embedded Chat Widget (JS)
 ├─ Formspree Contact Form
 └─ Static Pages
   │
   ▼
Render Cloud Service (faqb2cbot-api)
 ├─ FastAPI (faqb2cbot_app.py)
 ├─ LangChain + FAISS
 └─ OpenAI API (Chat + Embeddings)
   │
   ▼
Response JSON → Widget → User
```

### Component Summary
| Layer | Tool | Purpose |
|-------|------|----------|
| **Frontend** | WordPress.com Business | Public site, blog, landing page |
|  | Chat Widget (custom JS) | Sends user input → FastAPI |
|  | Formspree | Handles form submissions (no backend needed) |
| **API Layer** | Render Web Service | Hosts FastAPI app |
|  | FastAPI + Uvicorn | Core REST service |
|  | LangChain + FAISS | Retrieval pipeline |
|  | OpenAI API | Embeddings & LLM responses |
| **CI/CD** | GitHub → Render | Auto-deploy pipeline |
| **Infra** | render.yaml | Build, start, health check definitions |
| **Data** | FAISS Index | Vector storage for context retrieval |

---

## 📁 Repository Layout
```
.
├─ faqb2cbot_app.py          # FastAPI app (entrypoint)
├─ requirements.txt          # Python dependencies
├─ render.yaml               # Render IaC definition
├─ .github/
│  └─ workflows/
│     └─ render-deploy.yml   # CI/CD workflow (optional)
├─ .env.example              # Template for local .env
├─ docs/
│  └─ runbooks.md            # Ops/incident runbook (optional)
└─ README.md                 # This file
```

---

## 🧠 Core Application

**Entrypoint:**  
`faqb2cbot_app.py` exports a FastAPI instance named `app`.

**Endpoints**
| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/healthz` | GET | Health check for Render readiness |
| `/upload` | POST | Multipart form-data upload (requires `python-multipart`) |
| `/ask` | POST | Core RAG endpoint — retrieves context from FAISS and queries OpenAI |

**CORS:** Configured to allow requests from:  
`https://myaitoolset.com`, `https://www.myaitoolset.com`, and `http://localhost:3000`.

---

## 📦 Dependencies

**requirements.txt**
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-dotenv==1.0.1
pydantic==2.9.2
langchain==0.2.16
langchain-community==0.2.16
langchain-openai==0.1.23
python-multipart==0.0.9
faiss-cpu==1.12.0
```

> Note: `faiss-cpu==1.12.0` is confirmed available on Render.  
> If builds fail, clear Render cache and redeploy.

---

## ☁️ Render Deployment

**render.yaml**
```yaml
services:
  - type: web
    name: faqb2cbot-api
    env: python
    plan: free
    region: oregon
    buildCommand: |
      pip install --upgrade pip setuptools wheel
      pip install -r requirements.txt
    startCommand: uvicorn faqb2cbot_app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.9
      - key: ENV
        value: production
healthCheckPath: /healthz
```

**Environment Variables (Render Dashboard)**
```
OPENAI_API_KEY=<your key>
ENV=production
PYTHON_VERSION=3.11.9
```

**Health Check Path:** `/healthz`  
**Port:** `$PORT`  
**Host:** `0.0.0.0`

---

## 🧑‍💻 Local Development

**Setup**
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

**Run locally**
```bash
uvicorn faqb2cbot_app:app --reload --port 8000
```

**Test Endpoints**
```bash
curl http://localhost:8000/healthz
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"What stack do we use?"}'
```

---

## 🔁 CI/CD Pipeline

### Option A — Auto Deploy (Render Integration)
Render watches your GitHub repo → deploys automatically on push to `main`.

### Option B — GitHub Actions Workflow
`.github/workflows/render-deploy.yml`
```yaml
name: Deploy to Render

on:
  push:
    branches: [ "main" ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Trigger Render Deploy
        env:
          RENDER_API_KEY: ${{ secrets.RENDER_API_KEY }}
          SERVICE_ID: ${{ secrets.RENDER_SERVICE_ID }}
        run: |
          curl -s -X POST             -H "Accept: application/json"             -H "Authorization: Bearer $RENDER_API_KEY"             https://api.render.com/v1/services/$SERVICE_ID/deploys
```

---

## 🌐 Frontend Integration

- **Website:** Hosted on **WordPress.com Business** → [https://myaitoolset.com](https://myaitoolset.com)
- **Widget:** JavaScript chatbot widget connects to Render API (e.g. `https://faqb2cbot-api.onrender.com/ask`)
- **Forms:** Powered by **Formspree** for contact/intake
- **CORS:** Ensure both your main and www domains are allowed origins in FastAPI middleware

---

## 💾 Persistence

FAISS is currently in-memory.  
Options for persistence:
- Mount a **Render Disk** (e.g. `/var/data/faiss_index`)
- Save/load FAISS index via cloud storage (S3, GCS)

---

## 🔒 Security & Best Practices

- **Secrets:** only via Render environment variables  
- **CORS:** restrict to your real site domains  
- **Auth:** add API key validation for `/ask` if exposed publicly  
- **Rate limits:** implement lightweight limiter or proxy protection  
- **Model cost control:** use `gpt-4o-mini` for efficient RAG responses

---

## 🧾 Runbook (Common Incidents)

| Issue | Cause | Resolution |
|--------|--------|------------|
| Build error: Missing `python-multipart` | Missing dependency | Add to requirements, redeploy |
| Build error: FAISS wheel not found | Incompatible Python version | Use Python 3.11 + `faiss-cpu==1.12.0` |
| Health check fails | Wrong start command | Use `uvicorn faqb2cbot_app:app --host 0.0.0.0 --port $PORT` |
| CORS errors | Missing origin | Update CORS list in `faqb2cbot_app.py` |
| OpenAI unauthorized | Missing env var | Add `OPENAI_API_KEY` in Render settings |

---

## 📡 Monitoring

- **Render Logs:** runtime + build logs  
- **Health Endpoint:** `/healthz`  
- **Uptime Checks:** optional (UptimeRobot, BetterUptime)

---

## ✅ Summary

| Component | Description |
|------------|--------------|
| **Backend** | FastAPI + LangChain + FAISS |
| **Host** | Render |
| **Frontend** | WordPress.com + JS Widget |
| **Contact** | Formspree |
| **CI/CD** | GitHub → Render |
| **Secrets** | Render Env Vars |
| **Domain** | myaitoolset.com |
| **Primary App** | faqb2cbot_app.py |
