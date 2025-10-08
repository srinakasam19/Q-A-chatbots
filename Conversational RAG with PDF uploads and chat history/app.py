import streamlit as st  
from langchain.chains import create_history_aware_retriever, create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_chroma import Chroma

from langchain_community.chat_message_histories import ChatMessageHistory 
from langchain_core.chat_history import BaseChatMessageHistory 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PyPDFLoader 

import os 
from dotenv import load_dotenv 

# Load environment variables from .env file
load_dotenv() 

# Set HuggingFace token
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## --- Streamlit app setup --- ##
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload PDFs and chat with their content")

# Input field for Groq API key
api_key = st.text_input("Enter your Groq API key: ", type="password")

# Only proceed if the Groq API key is provided
if api_key:
    # Initialize Groq LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
    
    # Input for session ID to maintain chat history
    session_id = st.text_input("Session ID", value="default_session")
    
    # Initialize chat history store in Streamlit session_state
    if 'store' not in st.session_state:
        st.session_state.store = {}
        
    # File uploader for PDFs (supports multiple files)
    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    
    # Process the uploaded PDF files
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            # Save uploaded PDF temporarily
            temppdf = f"./trmp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name 
                
            # Load PDF and extract text
            loader = PyPDFLoader(temppdf)
            docs = loader.load() 
            documents.extend(docs)
    
        # Split the documents into smaller chunks for embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        
        # Create vector embeddings for the document chunks
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        # System prompt to convert user question into a standalone question considering chat history
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do not answer the question, "
            "just reformulate it if needed and otherwise return it as is"
        )
        
        # Prompt template for history-aware question reformulation
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create a retriever that takes chat history into account
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        ## --- Answer question prompt --- ##
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the retrieved context to answer the user's question. "
            "If you don't know the answer, say 'I don't know.' "
            "Keep your answer concise, maximum three sentences. "
            "\n\n"
            "{context}"
        )

        # Chat prompt template for generating answers using retrieved context
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # Chain for question-answering using the LLM and the QA prompt
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Create full RAG retrieval chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Function to get chat session history
        def get_session_hsitory(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory() 
            return st.session_state.store[session_id] 

        # Wrap RAG chain with message history to enable conversational context
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_hsitory,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        # Streamlit input for user's question
        user_input = st.text_input("Your Question:")
        if user_input:
            # Retrieve session history
            session_history = get_session_hsitory(session_id)
            
            # Invoke RAG chain to get answer
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                },
            )
            
            # Display full session store for debugging
            st.write(st.session_state.store)
            
            # Display assistant's answer
            st.write("Assistant:", response['answer'])

            # Display chat history for the session
            st.write("Chat History:", session_history.messages)

else:
    # Warning if Groq API key not provided
    st.warning("Please enter the Groq API key")






