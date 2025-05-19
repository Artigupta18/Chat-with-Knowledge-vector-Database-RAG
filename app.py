import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_community.vectorstores import FAISS
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from streamlit_chat import message


st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Chat with Knowledge vector Database")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "vector_db" not in st.session_state:
    st.session_state["vector_db"] = None
if "embedding_model" not in st.session_state:
    st.session_state["embedding_model"] = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    embeddings = st.session_state["embedding_model"]
    vector_db = FAISS.from_texts(chunks, embeddings)
    
    st.session_state["vector_db"] = vector_db
    st.success("Knowledge base updated successfully!")

def handle_query(user_query, api_key):
    if st.session_state["vector_db"] is None:
        return "Please upload documents first to build the knowledge base."
    
    docs = st.session_state["vector_db"].similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are an AI expert in construction and Briqko technology. 
    Use the following knowledge to answer user queries.
    
    Context: {context}
    User Question: {user_query}
    
    If context is insufficient, indicate that clearly.
    """
    
    api_key=st.secrets["API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    return response.text

st.sidebar.header("Upload Knowledge Base")
pdf_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if st.sidebar.button("Process PDFs") and pdf_files:
    with st.spinner("Processing PDFs..."):
        process_pdfs(pdf_files)

st.subheader("RAG Based Chatbot")
user_query = st.text_input("Ask a question")

if st.button("Send"):
    if user_query:
        with st.spinner("Thinking..."):
            response = handle_query(user_query, "api_key")
            st.session_state["chat_history"].append({"user": user_query, "bot": response})
    
for i, chat in enumerate(st.session_state["chat_history"]):
    message(chat["user"], is_user=True, key=f"user_{i}")
    message(chat["bot"], is_user=False, key=f"bot_{i}")


