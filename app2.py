import streamlit as st
import wikipedia
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os
load_dotenv()
api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')

embedder = SentenceTransformer("all-MiniLM-L6-v2")


llm = HuggingFaceHub(
    repo_id="deepseek-ai/deepseek-coder-6.7b-instruct",
    model_kwargs={"temperature": 0.3, "max_length": 300},
    huggingfacehub_api_token=api_token
)


index = faiss.IndexFlatL2(384)  
documents = []  


def retrieve_wikipedia_content(question):
    """Fetch Wikipedia content related to the question."""
    try:
        page = wikipedia.page(question, auto_suggest=True)
        content = page.content[:2000]  
        return content
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return "No relevant Wikipedia page found."


def update_vector_store(text):
    """Convert text into embeddings and store in FAISS."""
    global index, documents
    chunks = [text[i:i + 200] for i in range(0, len(text), 200)]  
    embeddings = embedder.encode(chunks) 
    index.add(np.array(embeddings).astype('float32'))  
    documents.extend(chunks) 


def retrieve_similar_text(question, top_k=3):
    """Find the most relevant Wikipedia text using FAISS."""
    query_embedding = embedder.encode([question]).astype('float32')
    _, indices = index.search(query_embedding, top_k)
    retrieved_text = " ".join([documents[idx] for idx in indices[0]])
    return retrieved_text


def generate_answer(question):
    """Generate an answer using retrieved Wikipedia content."""
    context = retrieve_similar_text(question)
    prompt = f"Based on the following Wikipedia information, answer the question:\n\n{context}\n\nQ: {question}\nA:"
    return llm.invoke(prompt)



st.title("📖 Wikipedia-Based RAG Q&A System")
st.write("Ask a question, and I'll fetch the best answer using Retrieval-Augmented Generation (RAG)!")

question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if question:
        with st.spinner("Fetching Wikipedia content..."):
            wiki_content = retrieve_wikipedia_content(question)
            update_vector_store(wiki_content)  # Store Wikipedia content
            response = generate_answer(question)
            st.write("### Answer:")
            st.success(response)
    else:
        st.warning("Please enter a question!")
