import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Define directories
pdf_directory = r"C:\Users\DELL\Desktop\data\backend\pdf2\temp"
persist_directory = "./data"

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load (or initialize) the persistent Chroma vector store
if os.path.exists(os.path.join(persist_directory, "chroma.sqlite")):
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
else:
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

# Document splitter
article_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
book_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Loop through each PDF and add incrementally
for filename in os.listdir(pdf_directory):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        print(f"Processing: {filename}")

        # Load text with PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        text_docs = loader.load()

        # Load tables with pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        text_table = "\n".join(
                            ["\t".join([str(cell) if cell is not None else "" for cell in row]) for row in table if row]
                        )
                        doc = Document(
                            page_content=text_table,
                            metadata={"source": filename, "page": i + 1, "type": "table"},
                        )
                        text_docs.append(doc)

        # Split documents according to the file type
        if filename.startswith("stat"):
            chunks = article_splitter.split_documents(text_docs)
        else:
            chunks = book_splitter.split_documents(text_docs)

        # Add chunks to vector store and persist
        vector_store.add_documents(chunks)
        vector_store.persist()
        print(f"Added and persisted: {filename}")
