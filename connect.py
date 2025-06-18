pipfrom langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize the embedding model (must be the same used during creation)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to the existing Chroma DB
vector_store = Chroma(persist_directory="data", embedding_function=embedding_model)

# Now you can query the vector store!
#______________________________________

query = "GDP?"

results = vector_store.similarity_search(query, k=3) # You can change k as needed

# Print the results
for i, result in enumerate(results):
    print(f"\nResult {i+1}:\n{result.page_content}")

