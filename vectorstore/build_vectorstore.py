import os
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Load OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Make sure it's set in your .env file.")

# ✅ Step 1: Load documents (change path as per your file)
loader = TextLoader("data/yourfile.txt")  # 🔁 Replace with your actual path
documents = loader.load()

# ✅ Step 2: Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# ✅ Step 3: Create Embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# ✅ Step 4: Create FAISS Vectorstore
vectorstore = FAISS.from_documents(docs, embedding_model)

# ✅ Step 5: Save Vectorstore
os.makedirs("vectorstore", exist_ok=True)
vectorstore.save_local("vectorstore")

# ✅ Save original docs for retrieval later (optional)
with open("vectorstore/index.pkl", "wb") as f:
    pickle.dump(docs, f)

print("✅ Vectorstore built and saved as: vectorstore/index.faiss and vectorstore/index.pkl")
