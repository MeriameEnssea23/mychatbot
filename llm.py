from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

key="my_key"

# Initialize the embedding model (must be the same used during creation)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to the existing Chroma DB
vector_store = Chroma(persist_directory= "./data", embedding_function=embedding_model)

# Retriever Configuration
retriever = vector_store.as_retriever(k=3)

best_prompt_kpi = ChatPromptTemplate.from_template("""
### Context
You are an intelligent assistant specialized in macroeconomic analysis.
You have access to a curated collection of macroeconomic KPI data extracted from reliable sources.

### Objective
Given a user's question and relevant KPI data, **adapt your response depending on the nature of the question**:

1. **Definition or Explanation Questions**:
   - If the question explicitly asks "What is...?" or "Define...", and if a definition exists in the context, display the **definition first**, clearly and completely.

2. **General Keyword Questions** (e.g. "GDP?", "Inflation?", "Unemployment?"):
   - If the question is short or vague and refers to a concept by name:
     - Start by displaying the **definition** of the term (if available in context).
     - Then, show **relevant data and statistics** related to that KPI from the context.
     - Finally, summarize any **insights, trends, or related KPIs**.

3. **Relationship Questions** (e.g., "How is inflation related to unemployment?"):
   - Focus on **correlations or relationships** between KPIs mentioned in the question.
   - Use context data to support the explanation.

4. **Trend Analysis or Evolution Questions** (e.g., "How has GDP evolved since 2020?"):
   - Focus on summarizing **trend patterns over time** for the KPIs involved.
   - Mention key years, values, and directions (increase/decrease).

5. **If the question is general but not keyword-only**, prioritize summarizing the **most relevant KPIs** and any **insights** based on available data.

### Tone
Use a professional, clear, and insightful tone. Make your answer accessible to both economists and general users.

### Response Requirements
Structure your response logically depending on the question:
- For **definition or keyword-type** questions:  
   1. Definition  
   2. Data/Statistics  
   3. Related insights or KPIs  
- Always cite specific data if available.
- Use only the provided context. Do not invent or assume information.
- cite the title of the article and the page used to provide the numerical information

<context>
{context}
</context>

Question: {input}

Guidelines:
- Use only the data and insights from the provided context.
- Do not generate any graphs or visual elements.
- Cite specific data when available.
- Format your response clearly, using bullet points or headings if helpful, Adjust the space between each title and its content. use 
""")

'''
query = "what is inflation targetting ?"

# Retrieve Relevant Documents
retriever_doc = retriever.get_relevant_documents(query=query)

#print("Retrieved Documents:", retriever_doc)

llm = ChatGroq(api_key=key, model='llama3-8b-8192')
# Create Document Chain with Selected Prompt
document_chain = create_stuff_documents_chain(llm, best_prompt_kpi)

# Execute Query and Measure Time

answer_response = document_chain.invoke({"input": query, "context": retriever_doc})

# Print Response and Execution Time
# ANSI escape code for green text
green = '\033[32m'
reset = '\033[0m'  # To reset the color back to default

print('##########################################################################################################################################################')
print('##########################################################################################################################################################')
print(f"{green}{answer_response}{reset}")
'''

def generate_response(user_query: str) -> str:
    retriever_doc = retriever.get_relevant_documents(query=user_query)
    
    
    llm = ChatGroq(api_key=key, model='llama3-8b-8192')
    document_chain = create_stuff_documents_chain(llm, best_prompt_kpi)
    raw_response=document_chain.invoke({"input":user_query,"context":retriever_doc})
    
    return {
        "question": user_query,
        "answer": raw_response  # ou transforme vers HTML ou JSON enrichi
    }

