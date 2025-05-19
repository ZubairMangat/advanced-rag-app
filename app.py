import os
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from PyPDF2 import PdfReader
import datetime

# Initialize SentenceTransformer and Groq client
retriever = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
api_key = st.secrets['key']
client = Groq(api_key=api_key)

# Global variables
documents = []
document_embeddings = []
chat_history = []

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Function to split long text into chunks
def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Update knowledge base
def update_knowledge_base(text, chunk_size):
    global documents, document_embeddings
    chunks = split_text_into_chunks(text, chunk_size)
    documents.extend(chunks)
    document_embeddings = retriever.encode(documents, convert_to_tensor=True)

# Retrieve relevant context
def retrieve(query, top_k):
    if not documents or len(document_embeddings) == 0:
        return None, None
    query_embedding = retriever.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, document_embeddings, top_k=top_k)
    top_chunks = [documents[hit['corpus_id']] for hit in hits[0]]
    return top_chunks, hits[0]

# Generate response using Groq's LLM
def generate_response(query, context):
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{query}"
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="📄 Advanced PDF Chatbot with Groq", layout="wide")
st.title("📄 Advanced PDF Chatbot using Groq + RAG")
st.markdown("Upload one or more PDF files and ask questions based on their content.")

# Sidebar Controls
with st.sidebar:
    st.header("🛠️ Settings")
    chunk_size = st.slider("🔹 Chunk Size (words)", 100, 1000, 500, step=100)
    top_k = st.slider("🔍 Top-K Chunks to Retrieve", 1, 5, 1)
    if st.button("🧹 Clear Knowledge Base"):
        documents.clear()
        document_embeddings.clear()
        st.success("Knowledge base cleared!")

# Upload multiple PDFs
uploaded_files = st.file_uploader("📤 Upload PDF file(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("📚 Reading and indexing PDF(s)..."):
        for uploaded_file in uploaded_files:
            text = extract_text_from_pdf(uploaded_file)
            if text:
                update_knowledge_base(text, chunk_size)
                st.success(f"✅ Processed: {uploaded_file.name}")
            else:
                st.error(f"❌ No text found in {uploaded_file.name}")

# Show all text chunks
if st.checkbox("📖 Show Extracted Chunks"):
    for i, chunk in enumerate(documents):
        st.markdown(f"**Chunk {i+1}:** {chunk[:300]}...")

# Question Input
question = st.text_input("💬 Ask your question here:")

if question:
    with st.spinner("🔎 Retrieving context..."):
        top_chunks, raw_hits = retrieve(question, top_k)
    if top_chunks:
        full_context = "\n---\n".join(top_chunks)
        with st.spinner("🤖 Generating answer..."):
            answer = generate_response(question, full_context)
            st.markdown(f"**Answer:** {answer}")
            with st.expander("🧠 View Retrieved Context"):
                st.text(full_context)
            chat_history.append(f"Q: {question}\nA: {answer}\n")
    else:
        st.warning("⚠️ No relevant context found.")

# Option to download chat history
if chat_history and st.button("📥 Download Chat History"):
    chat_text = "\n\n".join(chat_history)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button("Download Chat", chat_text, file_name=f"chat_history_{timestamp}.txt")

