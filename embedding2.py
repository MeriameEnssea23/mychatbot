import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

import pandas as pd
import numpy as np
import json
from mistralai import Mistral

my_api_key = "key"

# Load environment variables
load_dotenv()

# Define directories
pdf_directory = r"C:\Users\DELL\Desktop\essais\backend\doc_pdf\temp"
persist_directory = "./chroma_store"

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Detect header rows in table
def detect_header_row_count(table):
    count = 1
    for row in table[1:]:
        if row[0] is None or str(row[0]).strip().lower() == 'none' or str(row[0]).strip() == '':
            count += 1
        else:
            break
    return count

# Fill empty headers with previous values
def fill_none_with_previous(header_rows):
    filled = []
    for row in header_rows:
        new_row = []
        last_val = ''
        for cell in row:
            if cell is None or str(cell).strip().lower() == 'none' or str(cell).strip() == '':
                new_row.append(last_val)
            else:
                last_val = str(cell).strip()
                new_row.append(last_val)
        filled.append(new_row)
    return filled

# Initialize or load Chroma vector store
if os.path.exists(os.path.join(persist_directory, "chroma.sqlite")):
    print("Base Chroma existante trouvée, chargement...")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
else:
    print("Aucune base trouvée, création d'une nouvelle...")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

book_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
report_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

for filename in os.listdir(pdf_directory):
    pdf_path = os.path.join(pdf_directory, filename)

    if filename.startswith("stat"):
        print(f"Report Processing: {filename}")
        last_title = " "
        text_generated_doc = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                all_data_json = []
                tables = page.extract_tables()
                if not tables:
                    continue

                table_boxes = page.find_tables()
                top_side_table = [tbl.bbox[1] for tbl in table_boxes]

                titles = []
                words = page.extract_words()
                for word in words:
                    if "Table" in word["text"]:
                        line_top = word["top"]
                        line_words = [w["text"] for w in words if abs(w["top"] - line_top) < 10]
                        title_text = ' '.join(line_words)
                        titles.append({"text": title_text, "top": line_top})

                top_titles = [item["top"] for item in titles]

                offset = 0
                for index, table in enumerate(tables):
                    current_index = index - offset
                    if len(table) < 2:
                        top_side_table.remove(min(top_side_table))
                        offset += 1
                        continue

                    if len(titles) == 0:
                        actual_title = last_title
                    elif len(titles) == 1:
                        actual_title = titles[0]["text"]
                    else:
                        distance = top_side_table[current_index] - np.array(top_titles)
                        positives = [(i, val) for i, val in enumerate(distance) if val > 0]
                        min_index, _ = min(positives, key=lambda item: item[1])
                        actual_title = titles[min_index]["text"]

                    header_row_count = detect_header_row_count(table)
                    header_rows = fill_none_with_previous(table[:header_row_count])
                    full_headers = [' '.join(col).strip() for col in zip(*header_rows)]

                    df = pd.DataFrame(table[header_row_count:], columns=full_headers)
                    df = df.dropna(how='all')
                    data_json = df.to_dict(orient="records")
                    for row in data_json:
                        row["title"] = actual_title
                    all_data_json.extend(data_json)

                last_title = actual_title

                client = Mistral(api_key=my_api_key)
                messages = [
                    {"role": "system", "content": "You are an assistant that converts JSON tables into structured text."},
                    {"role": "user", "content": f"""Transform all these data into a smooth structured text, do not summarize or omit information. Replace the letter (e) before numerical value by the word estimated. If the year is not mentioned in the data do not make any assumptions about the year. If year is mentioned in the data use it every time it is mentioned. Do not condense the information. Generate a complete and accurate response without leaving anything out: {json.dumps(all_data_json, indent=1)}"""}
                ]

                try:
                    completion = client.chat.complete(
                        model="mistral-medium-latest",
                        messages=messages,
                        temperature=0.5,
                        max_tokens=8192,
                        top_p=1.0,
                        stream=False
                    )

                    texte_genere_par_llm_page = completion.choices[0].message.content
                    text_generated_doc.append(texte_genere_par_llm_page)

                except Exception as e:
                    print(f"Erreur lors de l'appel mistral sur la page {page.page_number} : {e}")

        print("End of conversion of report")

        docs = [
            Document(
                page_content=texte,
                metadata={"source": filename, "page": idx + 1, "type": "converted_table"}
            ) for idx, texte in enumerate(text_generated_doc)
        ]

        chunks = report_splitter.split_documents(docs)
        vector_store.add_documents(chunks)
        vector_store.persist()
        print(f"Added and persisted: {filename}")

    else:
        print(f"Book Processing: {filename}")
        loader = PyPDFLoader(pdf_path)
        text_docs = loader.load()
        chunks = book_splitter.split_documents(text_docs)
        vector_store.add_documents(chunks)
        vector_store.persist()
        print(f"Added and persisted: {filename}")
