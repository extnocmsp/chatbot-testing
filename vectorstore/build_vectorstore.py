import streamlit as st
import os
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# UI
st.title("Vectorstore Builder")
st.write("Upload your text file to build a vectorstore (index.faiss & index.pkl).")

uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

if uploaded_file is not None:
    with open("uploaded_file.txt", "wb") as f:
        f.write(uploaded_file.read())

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("❌ OpenAI API key not found. Check your .env or Streamlit secrets.")
    else:
        with st.spinner("Building vectorstore..."):
            loader = TextLoader("uploaded_file.txt")
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents(documents)

            embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectorstore = FAISS.from_documents(docs, embedding_model)

            os.makedirs("vectorstore", exist_ok=True)
            vectorstore.save_local("vectorstore")

            with open("vectorstore/index.pkl", "wb") as f:
                pickle.dump(docs, f)

        st.success("✅ Vectorstore generated successfully!")

