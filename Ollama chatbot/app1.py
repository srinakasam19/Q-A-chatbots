import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# Set LangChain API key if using LangChain tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

# --- Prompt template ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# --- Function to generate response ---
def generate_response(question, engine, temperature, max_tokens):
    llm = Ollama(model=engine, temperature=temperature)  # ✅ using temperature
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return str(answer)

# --- Sidebar settings ---
st.sidebar.title("Settings")
llm_model = st.sidebar.selectbox(
    "Select Ollama Model:",
    ["deepseek-r1:1.5b", "phi3:latest"]
)
temperature = st.sidebar.slider("Temperature (creativity)", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=350, value=150)

# --- User input ---
st.title("✅ Q&A Chatbot with Ollama")
st.write("Go ahead and ask any question:")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, llm_model, temperature, max_tokens)  # ✅ fixed call
    st.write(response)
else:
    st.write("Please provide your question above.")
