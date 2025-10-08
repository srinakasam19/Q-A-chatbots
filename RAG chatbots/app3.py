import os
import sys
import time
import streamlit as st

# -------------------------------
# Fix PyTorch & OpenMP conflicts
# -------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.modules["torch.classes"] = None

# -------------------------------
# Imports
# -------------------------------
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# -------------------------------
# Load API Keys
# -------------------------------
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# -------------------------------
# Initialize LLM
# -------------------------------
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# -------------------------------
# Define prompt
# -------------------------------
prompt = ChatPromptTemplate.from_template(
    """
Answer the question based on the provided context. 
Provide a detailed and well-explained response in simple terms.

<context>
{context}
</context>

Question: {input}
"""
)

# -------------------------------
# Function to create vector embeddings
# -------------------------------
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()

        pdf_folder = "C:/Users/srina/OneDrive/Documents/langchain projects/1-Q&A chatbot/RAG Q&A/research_papers"
        if not os.path.exists(pdf_folder):
            st.error(f"Folder not found: {pdf_folder}")
            return

        st.session_state.loader = PyPDFDirectoryLoader(pdf_folder)
        st.session_state.docs = st.session_state.loader.load()

        if not st.session_state.docs:
            st.error("No PDF files found in the folder.")
            return

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200
        )

        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs
        )

        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )

        st.success("✅ Vector Database is ready!")

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📘RAG Document Q&A Chatbot")

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("📂 Create Document Embedding"):
    create_vector_embedding()

# -------------------------------
# Querying
# -------------------------------
if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Create Document Embedding' first to load PDFs.")
    else:
        # -------------------------------
        # 1️⃣ Generate LLM Answer
        # -------------------------------
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 5})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        elapsed = time.process_time() - start

        st.subheader("🧠 Answer:")
        st.write(response.get('answer', "No answer generated."))
        st.write(f"⏱ Response time: {elapsed:.2f} sec")

        # -------------------------------
        # 2️⃣ Get documents for similarity search
        # -------------------------------
        st.subheader("📄 Document Similarity Search:")
        docs = retriever.get_relevant_documents(user_prompt)  # <-- This ensures we get actual docs
        if docs:
            for i, doc in enumerate(docs, start=1):
                doc_name = doc.metadata.get("source", f"Document {i}") if doc.metadata else f"Document {i}"
    

                chunks = doc.page_content.strip().split("\n\n")
                for chunk in chunks:
                    chunk = chunk.strip()
                    if chunk:
                        st.write(chunk)

                # Separator between documents
                st.markdown("---- ---- ---- ----")
        else:
            st.info("No matching text found in documents.")
