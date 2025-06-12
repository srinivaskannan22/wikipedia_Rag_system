📖 Wikipedia-Based RAG Q&A System
An AI-powered Q&A system using Retrieval-Augmented Generation (RAG) with Wikipedia, FAISS, and an LLM.

🚀 Overview
This project is a Wikipedia-based Q&A system that retrieves relevant Wikipedia content using FAISS and generates precise answers using a large language model (LLM).

📌 How It Works:
1️⃣ Retrieve Wikipedia content related to the user’s question
2️⃣ Chunk & store the text as embeddings in FAISS
3️⃣ Find the most relevant text using similarity search
4️⃣ Generate a natural-language answer using DeepSeek-Coder LLM

🔹 Features
✅ Wikipedia Content Retrieval – Automatically fetches relevant Wikipedia pages
✅ Efficient Search with FAISS – Uses vector similarity for fast retrieval
✅ Text Embeddings with Sentence Transformers – Converts text into numerical vectors
✅ LLM-Powered Answer Generation – Converts retrieved text into structured responses
✅ Streamlit UI – Interactive web app for easy Q&A

🛠 Tech Stack
Python
Streamlit
FAISS (Facebook AI Similarity Search)
Sentence Transformers (all-MiniLM-L6-v2)
Hugging Face (DeepSeek-Coder-6.7B-Instruct)
Wikipedia API
📌 Installation & Setup
🔹 1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/srinivaskannan22/wikipedia_Rag_system.git
🔹 2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
🔹 3. Set Up Hugging Face API Key
Create a .env file in the project folder and add your API key:

ini
Copy
Edit
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key
🔹 4. Run the Streamlit App
bash
Copy
Edit
streamlit run app.py
🎯 Example Usage
🔹 Input Question: "What is deep learning?"
🔹 Retrieved Wikipedia Context:

"Deep learning uses neural networks to improve performance. It requires large datasets for training..."
🔹 Generated Answer:
"Deep learning is a subset of machine learning that utilizes neural networks with multiple layers to analyze complex patterns in data."

📝 Future Enhancements
🚀 Expand to multiple knowledge sources beyond Wikipedia
🚀 Optimize retrieval for faster response times
🚀 Improve LLM prompts for more precise answers



