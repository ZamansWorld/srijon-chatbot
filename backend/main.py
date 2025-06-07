from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import os
import io
import fitz # PyMuPDF
from bs4 import BeautifulSoup # train URL
from pptx import Presentation # ppt files
import requests

app = FastAPI()

# Initialize base components
embedding_model = OpenAIEmbeddings(model="gpt-4-turbo")
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:3000"] if using a frontend server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ["OPENAI_API_KEY"] = "my-key"

# Initialize embedding and vector store
# embedding = OpenAIEmbeddings(model="gpt-4-turbo")
# embedding = OpenAIEmbeddings(model="text-embedding-3-small")
# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
VECTOR_DIR = "faiss_index"

vectorstore = FAISS.load_local(VECTOR_DIR, embeddings=embedding, allow_dangerous_deserialization=True) if os.path.exists("faiss_index") else FAISS.from_texts(["Initial dummy doc"], embedding)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

class Query(BaseModel):
    text: str

def extract_text_from_url(url: str) -> str:
    """Fetch and extract readable text from a URL"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    # Remove script/style
    for tag in soup(["script", "style"]):
        tag.extract()
    text = soup.get_text(separator=" ", strip=True)
    return text

def extract_text_from_pptx(file_path: str) -> str:
    prs = Presentation(file_path)
    all_text = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                all_text.append(shape.text)
    return "\n".join(all_text)

@app.post("/ask")
async def ask(query: Query):
    print(query)
    answer = qa_chain.invoke(query.text)
    print(answer)
    return {"response": answer}

@app.post("/train")
async def train(query: Query):
    vectorstore.add_texts([query.text])
    vectorstore.save_local(VECTOR_DIR)
    return {"status": "Training data added successfully."}

@app.post("/upload_txt")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        return {"message": "Only .txt files are supported."}

    contents = await file.read()
    text = contents.decode("utf-8")

    # Add to vector store and save
    vectorstore.add_texts([text])
    vectorstore.save_local(VECTOR_DIR)
    print(text)

    return {"message": f"File '{file.filename}' uploaded and embedded successfully."}

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {"message": "Only CSV files are supported."}

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Convert each row into a text block (customize this if needed)
        texts = []
        for _, row in df.iterrows():
            row_text = " | ".join(str(cell) for cell in row if pd.notnull(cell))
            texts.append(row_text)

        # Add to vector store
        vectorstore.add_texts(texts)
        vectorstore.save_local(VECTOR_DIR)

        return {"message": f"CSV file '{file.filename}' uploaded and trained successfully."}

    except Exception as e:
        return {"message": f"Error processing CSV: {str(e)}"}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"message": "Only PDF files are supported."}

    try:
        # Save file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Extract text from PDF using PyMuPDF
        doc = fitz.open(temp_path)
        all_text = "\n".join(page.get_text() for page in doc)
        doc.close()
        os.remove(temp_path)

        if not all_text.strip():
            return {"message": "No text found in the PDF."}

        # Add to vector store
        vectorstore.add_texts([all_text])
        vectorstore.save_local(VECTOR_DIR)

        return {"message": f"PDF file '{file.filename}' processed and embedded successfully."}

    except Exception as e:
        return {"message": f"Error processing PDF: {str(e)}"}

@app.post("/train_url")
async def train_url(url: str = Form(...)):
    try:
        text = extract_text_from_url(url)
        if not text.strip():
            return {"message": "No text found at the provided URL."}

        vectorstore.add_texts([text])
        vectorstore.save_local(VECTOR_DIR)

        return {"message": f"Content from URL '{url}' embedded and stored successfully."}
    except Exception as e:
        return {"message": f"Failed to process URL: {str(e)}"}

@app.post("/upload_pptx")
async def upload_pptx(file: UploadFile = File(...)):
    if not file.filename.endswith(".pptx"):
        return {"message": "Only .pptx files are supported."}

    try:
        # Save the file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Extract text
        text = extract_text_from_pptx(temp_path)
        os.remove(temp_path)

        if not text.strip():
            return {"message": "No text found in the PowerPoint file."}

        # Embed and store
        vectorstore.add_texts([text])
        vectorstore.save_local(VECTOR_DIR)

        return {"message": f"PowerPoint file '{file.filename}' processed and embedded successfully."}

    except Exception as e:
        return {"message": f"Error processing PowerPoint: {str(e)}"}
