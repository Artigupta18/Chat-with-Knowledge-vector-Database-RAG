# Chat-with-PDF
https://chat-with-briqko-ai-aqpvejqp6otpgngnzbxwr4.streamlit.app

The Construction Chatbot is an AI-powered assistant designed to provide expert insights into Briqko technology and its applications in the construction industry. The chatbot utilizes a vector knowledge database, allowing users to upload PDFs via the UI to enhance its knowledge base. It leverages LangGraph, Hugging Face, and Google Gemini AI to retrieve relevant information and generate intelligent responses.

# Features

Upload and process PDF documents to expand the chatbot's knowledge base.
Use FAISS vector database for efficient document retrieval.
Implement Hugging Face Embeddings for text similarity searches.
Query Google Gemini AI to generate responses.
Streamlit-based web UI for an interactive chat experience.
Persistent chat history for seamless interactions.

# Technologies Used

Python
Streamlit (Web UI)
FAISS (Vector database)
Hugging Face Embeddings (Text processing)
Google Gemini AI (LLM-powered responses)
PyPDF2 (PDF processing)
LangChain (Conversational AI framework)

# Setup Instructions

1. Install Dependencies

Create a virtual environment and install the required libraries.

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

2. Environment Variables

Create a .env file in the project directory and add your Google Gemini API key:

GOOGLE_API_KEY=your_google_gemini_api_key

3. Run the Chatbot

Start the Streamlit app by running:

streamlit run app.py

4. Upload Documents

Navigate to the sidebar and upload PDF documents.

Click "Process PDFs" to extract and store knowledge in the vector database.

5. Start Chatting

Type a question about Briqko and construction in the chat input field.

Click "Send" and receive an AI-generated response based on the knowledge base.

Code Breakdown

1. Initialize Streamlit UI

st.set_page_config(page_title="Briqko Construction Chatbot", layout="wide")
st.title("Chat with Briqko AI - Construction Expert 🚧")

2. Load and Process PDFs

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

3. Query Processing

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
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

4. Chat Interface

st.subheader("💬 Chat with Briqko AI")
user_query = st.text_input("Ask a question about Briqko & Construction:")
if st.button("Send"):
    if user_query:
        with st.spinner("Thinking..."):
            response = handle_query(user_query, api_key=os.getenv("GOOGLE_API_KEY"))
            st.session_state["chat_history"].append({"user": user_query, "bot": response})
for i, chat in enumerate(st.session_state["chat_history"]):
    message(chat["user"], is_user=True, key=f"user_{i}")
    message(chat["bot"], is_user=False, key=f"bot_{i}")

# Deployment

To deploy the chatbot on a cloud platform like Streamlit Community Cloud, follow these steps:
Push your project to GitHub.
Go to Streamlit Community Cloud and connect your repo.
Set the GOOGLE_API_KEY in the environment variables.
Deploy and start using the chatbot!

