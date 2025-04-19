import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_groq import ChatGroq  # type: ignore
from dotenv import load_dotenv # type: ignore
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# if "groq" not in st.secrets or "api_key" not in st.secrets["groq"]:
#     st.error("Groq API key not found! Please set it in Streamlit Secrets.")
#     st.stop()

# # Retrieve API key from Streamlit secrets
# api_key = st.secrets["groq"]["groq_api_key"]

# Define custom prompt
custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything out of the given context.
Question: {question} 
Context: {context} 
Answer:
"""

# Define paths
FAISS_DB_PATH = "vectorstore/db_faiss"
pdfs_directory = "pdfs/"

# Initialize Groq LLM (Change model as needed)
llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key = groq_api_key)  # Keeping Groq

# Step 1: Upload & Load PDF
def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

# Step 2: Create Chunks
def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Step 3: Setup Embeddings Model (Using Sentence Transformers)
def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# Step 4: Index Documents & Store in FAISS
def create_vector_store(db_faiss_path, text_chunks):
    faiss_db = FAISS.from_documents(text_chunks, get_embedding_model())
    faiss_db.save_local(db_faiss_path)
    return faiss_db

# Step 5: Retrieve Documents
def retrieve_docs(faiss_db, query):
    return faiss_db.similarity_search(query)

# Step 6: Generate Context
def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

# Step 7: Answer Query using LLM
def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})

# Streamlit UI
uploaded_file = st.file_uploader(
    "Upload PDF file that contains all the laws and regulations",
    type="pdf",
    accept_multiple_files=False
)

user_query = st.text_area("Enter your prompt: ", height=150, placeholder="Ask Anything!")

ask_question = st.button("Ask the AI Lawyer")

if ask_question:
    if uploaded_file and user_query:
        upload_pdf(uploaded_file)
        documents = load_pdf(pdfs_directory + uploaded_file.name)
        text_chunks = create_chunks(documents)
        faiss_db = create_vector_store(FAISS_DB_PATH, text_chunks)

        retrieved_docs = retrieve_docs(faiss_db, user_query)
        response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)

        st.chat_message("user").write(user_query)
        st.chat_message("AI Lawyer").write(response)

    else:
        st.error("Kindly upload a valid PDF file and/or ask a valid Question!")
