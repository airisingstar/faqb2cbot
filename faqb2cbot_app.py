import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load API key from .env or Streamlit secrets
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

# Page config
st.set_page_config(page_title="FAQ Assistant", page_icon="🦷")
st.title("🦷 FAQ Assistant")
st.markdown("Helping your customers get answers 24/7.")

# Load and process client file (PDF or TXT)
FILE_PATH = "client_faq.txt"  # Replace with "client_faq.pdf" if needed

if FILE_PATH.endswith(".pdf"):
    loader = PyPDFLoader(FILE_PATH)
elif FILE_PATH.endswith(".txt"):
    loader = TextLoader(FILE_PATH, encoding="utf-8")
else:
    st.error("Unsupported file type. Use .txt or .pdf.")
    st.stop()

# Load documents and split into chunks
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Embedding and FAISS vector DB (in-memory, Streamlit-friendly)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embedding=embeddings)
retriever = db.as_retriever()

# Chat memory and chain setup
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(temperature=0)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Session state for conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input box
user_input = st.text_input("Ask a question:", key="input")
if user_input:
    response = qa_chain.invoke({"question": user_input})
    answer = response["answer"]

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("🦷 FAQ Assistant", answer))

# Display full conversation
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")
