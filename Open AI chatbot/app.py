import streamlit as st
import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import sys

import io


# --- Load environment variables ---
load_dotenv()

# Set LangChain API key if using LangChain tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OPENAI"

# --- Prompt template ---
prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# --- Function to generate response ---
def generate_response(question, api_key, llm_model, temperature, max_tokens):
    if not api_key:
        return "❌ Please provide a valid OpenAI API key."
    
    llm = ChatOpenAI(
        model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return str(answer)  # Ensure Unicode-safe string

# --- Streamlit app ---
st.title("✅ Enhanced Q&A Chatbot with OpenAI")

# --- Sidebar settings ---
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
llm_model = st.sidebar.selectbox(
    "Select OpenAI Model:",
    ["gpt-4.1-2025-04-14", "gpt-4o-2024-08-06", "gpt-4.1-mini"]
)
temperature = st.sidebar.slider("Temperature (creativity)", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=350, value=150)

# --- User input ---
st.write("Go ahead and ask any question:")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, api_key, llm_model, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide your question above.")
