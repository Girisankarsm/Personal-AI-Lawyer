from vector_database import retrieve_docs, get_context, faiss_db  # noqa: F401
from langchain_core.prompts import ChatPromptTemplate

# LLM is owned by Agent 1 / app.py (prefer free local Ollama).
# Keep a soft import so retrieval works even without Groq.
try:
    from langchain_groq import ChatGroq
    from dotenv import load_dotenv
    import os

    load_dotenv()
    _key = os.getenv("GROQ_API_KEY")
    llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=_key) if _key else None
except Exception:  # noqa: BLE001
    llm_model = None

custom_prompt_template = """
You are an educational legal research assistant. Use ONLY the context below.
If the context is insufficient, say you do not know. Do not invent citations.
This is not legal advice.

Question: {question}
Context: {context}
Answer:
"""


def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})
