from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from dotenv import load_dotenv
import os

# Load .env file from current directory
load_dotenv()

# Make sure the environment variable is available to LangChain
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("Missing OPENAI_API_KEY in .env")

os.environ["OPENAI_API_KEY"] = openai_key

# Load and split document
loader = TextLoader("faq.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)

# Embed and index the documents
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

# Load LLM and chain
llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff")

# Initialize FastAPI app
app = FastAPI()

# CORS for your frontend widget
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to specific domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class QuestionRequest(BaseModel):
    question: str

# Route to ask questions
@app.post("/ask")
async def ask(req: QuestionRequest):
    result = chain.run(
        input_documents=db.similarity_search(req.question),
        question=req.question
    )
    return {"answer": result}
