import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_community.vectorstores import FAISS
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

try:
    api_key = st.secrets["API"]["API_KEY"]
    st.write("API Key found:", api_key[:5] + "...")
except Exception as e:
    st.write("Error:", e)

# Initialize Streamlit UI
st.set_page_config(page_title="Briqko Construction Chatbot", layout="wide")
st.title("Chat with Briqko AI - Construction Expert 🚧")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "vector_db" not in st.session_state:
    st.session_state["vector_db"] = None
if "embedding_model" not in st.session_state:
    st.session_state["embedding_model"] = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to process PDFs and update knowledge base
def process_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    # Convert chunks into vector embeddings
    embeddings = st.session_state["embedding_model"]
    vector_db = FAISS.from_texts(chunks, embeddings)
    
    # Store vector DB in session
    st.session_state["vector_db"] = vector_db
    st.success("Knowledge base updated successfully!")

# Function to handle user query
def handle_query(user_query, api_key):
    if st.session_state["vector_db"] is None:
        return "Please upload documents first to build the knowledge base."
    
    # Retrieve relevant context
    docs = st.session_state["vector_db"].similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Define prompt template
    prompt = f"""
    You are an AI expert in construction and Briqko technology. 
    Use the following knowledge to answer user queries.
    
    Context: {context}
    User Question: {user_query}
    
    If context is insufficient, indicate that clearly.
    """
    
    # Query Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    return response.text

# Sidebar for uploading PDFs
st.sidebar.header("📂 Upload Knowledge Base")
pdf_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if st.sidebar.button("Process PDFs") and pdf_files:
    with st.spinner("Processing PDFs..."):
        process_pdfs(pdf_files)

# Chat interface
st.subheader("💬 Chat with Briqko AI")
user_query = st.text_input("Ask a question about Briqko & Construction:")

if st.button("Send"):
    if user_query:
        with st.spinner("Thinking..."):
            response = handle_query(user_query, "api_key")
            st.session_state["chat_history"].append({"user": user_query, "bot": response})
    
# Display chat history (WhatsApp-style)
for i, chat in enumerate(st.session_state["chat_history"]):
    message(chat["user"], is_user=True, key=f"user_{i}")
    message(chat["bot"], is_user=False, key=f"bot_{i}")


