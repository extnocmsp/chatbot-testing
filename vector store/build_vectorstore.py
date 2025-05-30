import os
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load your OpenAI API key from .env or Streamlit secrets
openai_api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Load documents
loader = TextLoader("your_documents_folder/yourfile.txt")  # You can use PDFLoader, CSVLoader, etc.
documents = loader.load()

# Step 2: Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Step 3: Embed using OpenAI
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Step 4: Create FAISS vectorstore
vectorstore = FAISS.from_documents(docs, embedding_model)

# Step 5: Save index and metadata
vectorstore.save_local("vectorstore")
with open("vectorstore/index.pkl", "wb") as f:
    pickle.dump(docs, f)

print("✅ Vectorstore built and saved as index.faiss and index.pkl")

