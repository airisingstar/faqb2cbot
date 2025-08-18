import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Page config
st.set_page_config(page_title="FAQ Assistant", page_icon="🦷")
st.title("🦷 FAQ Assistant")
st.markdown("Helping your customers get answers 24/7.")

# Load and process your static file (change path as needed)
FILE_PATH = "client_faq.txt"  # or use "client_faq.pdf"

# Choose loader based on file type
if FILE_PATH.endswith(".pdf"):
    loader = PyPDFLoader(FILE_PATH)
elif FILE_PATH.endswith(".txt"):
    loader = TextLoader(FILE_PATH, encoding="utf-8")
else:
    st.error("Unsupported file type. Use .txt or .pdf.")
    st.stop()

# Load documents and create chunks
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Embedding and vector DB
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory=None  # Force in-memory, ephemeral usage
)
retriever = db.as_retriever()

# Chat memory and chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(temperature=0)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question:", key="input")
if user_input:
    response = qa_chain.invoke({"question": user_input})
    answer = response["answer"]

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("🦷 FAQ Assistant", answer))

# Display conversation
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")
